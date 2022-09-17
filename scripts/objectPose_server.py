#!/usr/bin/env python3

from __future__ import print_function

import multiprocessing

from odl.srv import ObjectPoseService,ObjectPoseServiceResponse
from odl.msg import ObjectPose
from odl.msg import ObjectId
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
import rospy

# import epos inference dependencies
import os
import os.path
import time
import numpy as np
import cv2
import tensorflow as tf
import pyprogressivex
import bop_renderer
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import transform
from bop_toolkit_lib import visualization
from epos_lib import common
from epos_lib import config
from epos_lib import corresp
from epos_lib import datagen
from epos_lib import misc
from epos_lib import model
from epos_lib import vis
#import multiprocessing
#from numba import cuda
import gc


# import createTfRecords dependencies
import io
import random
from functools import partial
from PIL import Image as ImagePIL
from epos_lib import tfrecord

# import create example list dependencies
import glob

# save images
import imageio
import png

# health msg
import threading
from odl.msg import SystemHealth

# Class for epos functionality (create input)
class createExampleListCls:
    def __init__(self):
        self.dataset = "carObj1"
        self.split = "test"
        self.split_type = None
        self.scene_ids = None
        self.targets_filename = None#"test_targets_car.json"
        self.output_dir = os.path.join(config.TF_DATA_PATH, 'example_lists')

        tf.logging.set_verbosity(tf.logging.ERROR) # INFO) TODO - avoid too much logging (INFO) for infer

        if self.scene_ids is not None and self.targets_filename is not None:
            raise ValueError(
                'Only up to one of scene_ids and targets_filename can be specified.')

        # Load dataset parameters.
        self.dp_split = dataset_params.get_split_params(
            config.BOP_PATH, self.dataset, self.split, self.split_type)

        self.output_suffix = None

    def createExampleList(self):
        if self.targets_filename:
            self.output_suffix = 'targets'
            test_targets = inout.load_json(
                os.path.join(config.BOP_PATH, self.dataset, self.targets_filename))
            example_list = []
            for trg in test_targets:
                example = {'scene_id': trg['scene_id'], 'im_id': trg['im_id']}
                if example not in example_list:
                    example_list.append(example)

        else:
            if self.scene_ids is None:
                self.scene_ids = dataset_params.get_present_scene_ids(self.dp_split)
            else:
                self.scene_ids = list(map(int, self.scene_ids))
                self.output_suffix = 'scenes-' + '-'.join(
                    map(lambda x: '{:01d}'.format(x), self.scene_ids))

            tf.logging.info('Collecting examples...')
            example_list = []
            for scene_id in self.scene_ids:
                scene_gt_fpath = self.dp_split['scene_gt_tpath'].format(scene_id=scene_id)

                # TODO car
                scene_ids_fromFolder = os.path.split(scene_gt_fpath)
                imagesListed = glob.glob(os.path.join(str(scene_ids_fromFolder[0]) + '/rgb', '*.png'))
                im_ids = []
                for i in range(0, len(imagesListed)):
                    imagesListed_withoutPath = (imagesListed[i].rsplit('/'))[-1]  # cut off path
                    imagesListed_withoutPathAndFormat = (imagesListed_withoutPath.split('.'))[0]  # cut off format
                    im_ids.append(int(imagesListed_withoutPathAndFormat))
                im_ids = sorted(im_ids)
                # print(im_ids)
                # TODO car

                # im_ids = inout.load_scene_gt(scene_gt_fpath).keys() # TODO car. this is the default for im_ids. Use the above instead to read images directly from the images folder, without needing gt
                for im_id in sorted(im_ids):
                    example_list.append({'scene_id': scene_id, 'im_id': im_id})

        tf.logging.info('Collected {} examples.'.format(len(example_list)))
        assert (len(example_list) > 0)

        split_name = self.split
        if self.split_type is not None:
            split_name += '-' + self.split_type

        if self.output_suffix is not None:
            self.output_suffix = '_' + self.output_suffix
        else:
            self.output_suffix = ''

        output_fname = '{}_{}{}_examples.txt'.format(
            self.dataset, split_name, self.output_suffix)
        output_fpath = os.path.join(self.output_dir, output_fname)

        tf.logging.info('Saving the list to: {}'.format(output_fpath))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        tfrecord.save_example_list(output_fpath, example_list)

# Class for epos functionality (create input)
class createTfRecordsCls:
    def is_pt_in_im(pt, im_size):
        return 0 <= pt[0] < im_size[0] and 0 <= pt[1] < im_size[1]

    def encode_image(im, format):
        with io.BytesIO() as output:
            if format.lower() in ['jpg', 'jpeg']:
                ImagePIL.fromarray(im).save(output, format='JPEG', subsampling=0, quality=95)
            else:
                ImagePIL.fromarray(im).save(output, format=format.upper())
            im_encoded = output.getvalue()
        return im_encoded

    def create_tf_example(self,
            example, dp_split, scene_camera, scene_gt=None, scene_gt_info=None):

        scene_id = example['scene_id']
        im_id = example['im_id']
        width = dp_split['im_size'][0]
        height = dp_split['im_size'][1]
        K = scene_camera[scene_id][im_id]['cam_K']

        gts = None
        gts_info = None
        mask_visib_fpaths = None
        if self.add_gt:
            gts = scene_gt[scene_id][im_id]
            gts_info = scene_gt_info[scene_id][im_id]

            # Collect paths to object masks.
            mask_visib_fpaths = []
            for gt_id in range(len(gts)):
                mask_visib_fpaths.append(dp_split['mask_visib_tpath'].format(
                    scene_id=scene_id, im_id=im_id, gt_id=gt_id))

        # RGB image.
        im_path = None
        rgb_encoded = None
        if 'rgb' in dp_split['im_modalities']:

            # Absolute path to the RGB image.
            im_path = dp_split['rgb_tpath'].format(scene_id=scene_id, im_id=im_id)

            # Determine the format of the RGB image.
            rgb_format_in = im_path.split('.')[-1]
            if rgb_format_in in ['jpg', 'jpeg']:
                rgb_format_in = 'jpg'

            # Load the RGB image.
            if rgb_format_in == self.rgb_format:
                with tf.gfile.GFile(im_path, 'rb') as fid:
                    rgb_encoded = fid.read()
            else:
                rgb = inout.load_im(im_path)
                rgb_encoded = self.encode_image(rgb, self.rgb_format)

        # Grayscale image.
        elif 'gray' in dp_split['im_modalities']:

            # Absolute path to the grayscale image.
            im_path = dp_split['gray_tpath'].format(scene_id=scene_id, im_id=im_id)

            # Load the grayscale image and duplicate the channel.
            gray = inout.load_im(im_path)
            rgb = np.dstack([gray, gray, gray])
            rgb_encoded = self.encode_image(rgb, self.rgb_format)

        # Path of the image relative to BOP_PATH.
        im_path_rel = im_path.split(config.BOP_PATH)[1]
        im_path_rel_encoded = im_path_rel.encode('utf8')

        # Collect ground-truth information about the annotated object instances.
        pose_q1, pose_q2, pose_q3, pose_q4 = [], [], [], []
        pose_t1, pose_t2, pose_t3, t4 = [], [], [], []
        obj_ids = []
        obj_ids_txt = []
        obj_visibilities = []
        masks_visib_encoded = []
        if self.add_gt:
            for gt_id, gt in enumerate(gts):
                # Orientation of the object instance.
                R = np.eye(4)
                R[:3, :3] = gt['cam_R_m2c']
                q = transform.quaternion_from_matrix(R)
                pose_q1.append(q[0])
                pose_q2.append(q[1])
                pose_q3.append(q[2])
                pose_q4.append(q[3])

                # Translation of the object instance.
                t = gt['cam_t_m2c'].flatten()
                pose_t1.append(t[0])
                pose_t2.append(t[1])
                pose_t3.append(t[2])

                obj_ids_txt.append(str(gt['obj_id']).encode('utf8'))
                obj_ids.append(int(gt['obj_id']))
                obj_visibilities.append(float(gts_info[gt_id]['visib_fract']))

                # Mask of the visible part of the object instance.
                with tf.gfile.GFile(mask_visib_fpaths[gt_id], 'rb') as fid:
                    mask_visib_encoded_png = fid.read()
                    masks_visib_encoded.append(mask_visib_encoded_png)

        # Intrinsic camera parameters.
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        # TF Example.
        feature = {
            'image/scene_id': tfrecord.int64_list_feature(scene_id),
            'image/im_id': tfrecord.int64_list_feature(im_id),
            'image/path': tfrecord.bytes_list_feature(im_path_rel_encoded),
            'image/encoded': tfrecord.bytes_list_feature(rgb_encoded),
            'image/width': tfrecord.int64_list_feature(width),
            'image/height': tfrecord.int64_list_feature(height),
            'image/channels': tfrecord.int64_list_feature(3),
            'image/camera/fx': tfrecord.float_list_feature([fx]),
            'image/camera/fy': tfrecord.float_list_feature([fy]),
            'image/camera/cx': tfrecord.float_list_feature([cx]),
            'image/camera/cy': tfrecord.float_list_feature([cy]),
            'image/object/id': tfrecord.int64_list_feature(obj_ids),
            'image/object/visibility': tfrecord.float_list_feature(obj_visibilities),
            'image/object/pose/q1': tfrecord.float_list_feature(pose_q1),
            'image/object/pose/q2': tfrecord.float_list_feature(pose_q2),
            'image/object/pose/q3': tfrecord.float_list_feature(pose_q3),
            'image/object/pose/q4': tfrecord.float_list_feature(pose_q4),
            'image/object/pose/t1': tfrecord.float_list_feature(pose_t1),
            'image/object/pose/t2': tfrecord.float_list_feature(pose_t2),
            'image/object/pose/t3': tfrecord.float_list_feature(pose_t3),
            'image/object/mask': tfrecord.bytes_list_feature(masks_visib_encoded),
        }
        tf_example = tf.train.Example(features=tf.train.Features(feature=feature))

        res = tf_example.SerializeToString()
        return res, example

    def __init__(self):
        #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        self.examples_filename = "carObj1_test_examples.txt"
        self.dataset = "carObj1"
        self.split = "test"
        self.split_type = "primesense"
        self.add_gt=True
        self.shuffle = False
        self.rgb_format = "png"
        self.output_dir = os.path.join(config.TF_DATA_PATH)

        tf.logging.set_verbosity(tf.logging.ERROR)#INFO)

        # Load the list examples.
        examples_path = os.path.join(
            config.TF_DATA_PATH, 'example_lists', self.examples_filename)
        tf.logging.info('Loading a list of examples from: {}'.format(examples_path))
        self.examples_list = tfrecord.load_example_list(examples_path)

        # Load dataset parameters.
        self.dp_split = dataset_params.get_split_params(
            config.BOP_PATH, self.dataset, self.split, self.split_type)

        # Pre-load camera parameters and ground-truth annotations.
        self.scene_gt = {}
        self.scene_gt_info = {}
        self.scene_camera = {}
        scene_ids = set([e['scene_id'] for e in self.examples_list])
        for scene_id in scene_ids:

            self.scene_camera[scene_id] = inout.load_scene_camera(
                self.dp_split['scene_camera_tpath'].format(scene_id=scene_id))

            if self.add_gt:
                self.scene_gt[scene_id] = inout.load_scene_gt(
                    self.dp_split['scene_gt_tpath'].format(scene_id=scene_id))
                self.scene_gt_info[scene_id] = inout.load_json(
                    self.dp_split['scene_gt_info_tpath'].format(scene_id=scene_id),
                    keys_to_int=True)

        # Check the name of the file with examples.
        self.examples_end = '_examples.txt'
        if not self.examples_filename.endswith(self.examples_end):
            raise ValueError(
                'Name of the file with examples must end with {}.'.format(self.examples_end))

    def runCreateTfRecords(self):
        # Prepare writer of the TFRecord file.
        output_name = self.examples_filename.split(self.examples_end)[0]
        output_path = os.path.join(self.output_dir, output_name + '.tfrecord')
        writer = tf.python_io.TFRecordWriter(output_path)
        tf.logging.info('File to be created: {}'.format(output_path))

        # Optionally shuffle the examples.
        if self.shuffle:
            random.shuffle(self.examples_list)

        # Write the examples to the TFRecord file.
        w_start_t = time.time()

        create_tf_example_partial = partial(
            self.create_tf_example,
            dp_split=self.dp_split,
            scene_camera=self.scene_camera,
            scene_gt=self.scene_gt,
            scene_gt_info=self.scene_gt_info)

        print("examples list: ", self.examples_list)
        for example_id, example in enumerate(self.examples_list):
            if example_id % 50 == 0:
                tf.logging.info('Processing example {}/{}'.format(
                    example_id + 1, len(self.examples_list)))

            tf_example, _ = create_tf_example_partial(example)
            writer.write(tf_example)

        # Close the writer.
        writer.close()

        w_total_t = time.time() - w_start_t
        tf.logging.info('Writing took {} s.'.format(w_total_t))

# Class for actual epos functionality
class eposInfer:
    def __init__(self):
        self.master = ''
        self.cpu_only =False
        self.task_type = common.LOCALIZATION #localization
        self.infer_tfrecord_names = ['carObj1_test']
        self.infer_max_height_before_crop = 720
        self.infer_crop_size = [1280, 720]
        self.checkpoint_name = None
        self.project_to_surface = False
        self.save_estimates = True
        self.save_corresp = False
        self.infer_name = None

        # Pose fitting parameters.
        self.fitting_method = common.PROGRESSIVE_X #progressive_x
        self.inlier_thresh = 4.0
        self.neighbour_max_dist = 20.0
        self.min_hypothesis_quality = 0.5
        self.required_progx_confidence = 0.5
        self.required_ransac_confidence = 1.0
        self.min_triangle_area = 0.0
        self.use_prosac = False
        self.max_model_number_for_pearl = 5
        self.spatial_coherence_weight = 0.1
        self.scaling_from_millimeters = 0.1
        self.max_tanimoto_similarity = 0.9
        self.max_correspondences = None
        self.max_instances_to_fit = None
        self.max_fitting_iterations = 400

        # Visualization parameters.
        self.vis = True # TODO - deactivate visualizations
        self.vis_gt_poses = False
        self.vis_pred_poses = True
        self.vis_gt_obj_labels = False
        self.vis_pred_obj_labels = False
        self.vis_pred_obj_confs = False
        self.vis_gt_frag_fields = False
        self.vis_pred_frag_fields = False

        self.dataset = "carObj1"
        self.model_variant = "xception_65"
        self.atrous_rates = [12, 24, 36]
        self.encoder_output_stride = 8
        self.decoder_output_stride = [4]
        self.upsample_logits = False
        self.frag_seg_agnostic = False
        self.frag_loc_agnostic = False
        self.num_frags = 64
        self.corr_min_obj_conf = 0.1
        self.corr_min_frag_rel_conf = 0.5
        self.corr_project_to_model = False
        self.model = "extend"  # TODO

        self.frag_cls_agnostic = False
        self.image_pyramid = None
        self.samples = None
        #self.posesPredicted = dict() # TODO

        tf.logging.set_verbosity(tf.logging.ERROR)#INFO)

        # Model folder.
        self.model_dir = os.path.join(config.TF_MODELS_PATH, self.model)

        # Update flags with parameters loaded from the model folder.
        common.update_flags(os.path.join(self.model_dir, common.PARAMS_FILENAME))

        # Print the flag values.
        #common.print_flags()

        # Folder from which the latest model checkpoint will be loaded.
        self.checkpoint_dir = os.path.join(self.model_dir, 'train')

        # Folder for the inference output.
        self.infer_dir = os.path.join(self.model_dir, 'infer')
        tf.gfile.MakeDirs(self.infer_dir)

        # Folder for the visualization output. NOT NEEDED FOR INFER
        self.vis_dir = os.path.join(self.model_dir, 'vis')
        tf.gfile.MakeDirs(self.vis_dir)

    def predictPose(self):
        # TFRecord files used for training.
        tfrecord_names = self.infer_tfrecord_names
        if not isinstance(self.infer_tfrecord_names, list):
            tfrecord_names = [self.infer_tfrecord_names]

        if self.upsample_logits:
            # The stride is 1 if the logits are upsampled to the input resolution.
            self.output_stride = 1
        else:
            assert (len(self.decoder_output_stride) == 1)
            self.output_stride = self.decoder_output_stride[0]

        with tf.Graph().as_default():

            self.return_gt_orig = np.any([
                self.task_type == common.LOCALIZATION,
                self.vis_gt_poses])

            self.return_gt_maps = np.any([
                self.vis_pred_obj_labels,
                self.vis_pred_obj_confs,
                self.vis_pred_frag_fields])

            # Dataset provider. # TODO dont know to where htis is
            dataset = datagen.Dataset(
                dataset_name=self.dataset,
                tfrecord_names=tfrecord_names,
                model_dir=self.model_dir,
                model_variant=self.model_variant,
                batch_size=1,
                max_height_before_crop=self.infer_max_height_before_crop,
                crop_size=list(map(int, self.infer_crop_size)),
                num_frags=self.num_frags,
                min_visib_fract=None,
                gt_knn_frags=1,
                output_stride=self.output_stride,
                is_training=False,
                return_gt_orig=self.return_gt_orig,
                return_gt_maps=self.return_gt_maps,
                should_shuffle=False,
                should_repeat=False,
                prepare_for_projection=self.project_to_surface,
                data_augmentations=None)

            # Initialize a renderer for visualization.
            renderer = None
            if self.vis_gt_poses or self.vis_pred_poses:
                tf.logging.info('Initializing renderer for visualization...')

                renderer = bop_renderer.Renderer()
                renderer.init(dataset.crop_size[0], dataset.crop_size[1])

                model_type_vis = 'eval'
                dp_model = dataset_params.get_model_params(
                    config.BOP_PATH, dataset.dataset_name, model_type=model_type_vis)
                for obj_id in dp_model['obj_ids']:
                    path = dp_model['model_tpath'].format(obj_id=obj_id)
                    renderer.add_object(obj_id, path)

                tf.logging.info('Renderer initialized.')

            # Inputs.
            self.samples = dataset.get_one_shot_iterator().get_next()

            # A map from output type to the number of associated channels.
            outputs_to_num_channels = common.get_outputs_to_num_channels(
                dataset.num_objs, dataset.model_store.num_frags)

            # Options of the neural network model.
            model_options = common.ModelOptions(
                outputs_to_num_channels=outputs_to_num_channels,
                crop_size=list(map(int, self.infer_crop_size)),
                atrous_rates=self.atrous_rates,
                encoder_output_stride=self.encoder_output_stride)

            # Construct the inference graph.
            predictions = model.predict(
                images=self.samples[common.IMAGE],
                model_options=model_options,
                upsample_logits=self.upsample_logits,
                image_pyramid=self.image_pyramid,
                num_objs=dataset.num_objs,
                num_frags=dataset.num_frags,
                frag_cls_agnostic=self.frag_cls_agnostic,
                frag_loc_agnostic=self.frag_loc_agnostic)

            # Get path to the model checkpoint.
            if self.checkpoint_name is None:
                self.checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_dir)
            else:
                self.checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name)

            time_str = time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime())
            tf.logging.info('Starting inference at: {}'.format(time_str))
            tf.logging.info('Inference with model: {}'.format(self.checkpoint_path))

            # Scaffold for initialization.
            self.scaffold = tf.train.Scaffold(
                init_op=tf.global_variables_initializer(),
                saver=tf.train.Saver(var_list=misc.get_variable_dict()))

            # TensorFlow configuration.
            if self.cpu_only:
                self.tf_config = tf.ConfigProto(device_count={'GPU': 0})
            else:
                self.tf_config = tf.ConfigProto()
                self.tf_config.gpu_options.allow_growth = False  # Only necessary GPU memory. # lele
                # tf_config.gpu_options.allow_growth = False

            # Nodes that can use multiple threads to parallelize their execution will
            # schedule the individual pieces into this pool.
            self.tf_config.intra_op_parallelism_threads = 10

            # All ready nodes are scheduled in this pool.
            self.tf_config.inter_op_parallelism_threads = 10

            self.poses_all = []
            self.first_im_poses_num = 0

            self.session_creator = tf.train.ChiefSessionCreator(
                config=self.tf_config,
                scaffold=self.scaffold,
                master=self.master,
                checkpoint_filename_with_path=self.checkpoint_path)
            with tf.train.MonitoredSession(
                    session_creator=self.session_creator, hooks=None) as sess:

                im_ind = 0
                while not sess.should_stop():

                    # Estimate object poses for the current image.
                    poses, run_times = self.process_image(
                        sess=sess,
                        samples=self.samples,
                        predictions=predictions,
                        im_ind=im_ind,
                        crop_size=dataset.crop_size,
                        output_scale=(1.0 / self.output_stride),
                        model_store=dataset.model_store,
                        renderer=renderer,
                        task_type=self.task_type,
                        infer_name=self.infer_name,
                        infer_dir=self.infer_dir,
                        vis_dir=self.vis_dir)

                    # Note that the first image takes longer time (because of TF init).
                    tf.logging.info(
                        'Image: {}, prediction: {:.3f}, establish_corr: {:.3f}, '
                        'fitting: {:.3f}, total time: {:.3f}'.format(
                            im_ind, run_times['prediction'], run_times['establish_corr'],
                            run_times['fitting'], run_times['total']))

                    self.poses_all += poses
                    if im_ind == 0:
                        first_im_poses_num = len(poses)
                    im_ind += 1

            # Set the time of pose estimates from the first image to the average time.
            # Tensorflow takes a long time on the first image (because of init).
            time_avg = 0.0
            for pose in self.poses_all:
                time_avg += pose['time']
            if len(self.poses_all) > 0:
                time_avg /= float((len(self.poses_all)))
            for i in range(first_im_poses_num):
                self.poses_all[i]['time'] = time_avg

            # Save the estimated poses in the BOP format:
            # https://bop.felk.cvut.cz/challenges/bop-challenge-2020/#formatofresults
            if self.save_estimates:
                suffix = ''
                if self.infer_name is not None:
                    suffix = '_{}'.format(self.infer_name)
                poses_path = os.path.join(
                    self.infer_dir, 'estimated-poses{}.csv'.format(suffix))
                tf.logging.info('Saving estimated poses to: {}'.format(poses_path))
                inout.save_bop_results(poses_path, self.poses_all, version='bop19')

            time_str = time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime())
            tf.logging.info('Finished inference at: {}'.format(time_str))
            return self.poses_all

    def visualize(self,
            samples, predictions, pred_poses, im_ind, crop_size, output_scale,
            model_store, renderer, vis_dir):
        """Visualizes estimates from one image.

        Args:
          samples: Dictionary with input data.
          predictions: Dictionary with predictions.
          pred_poses: Predicted poses.
          im_ind: Image index.
          crop_size: Image crop size (width, height).
          output_scale: Scale of the model output w.r.t. the input (output / input).
          model_store: Store for 3D object models of class ObjectModelStore.
          renderer: Renderer of class bop_renderer.Renderer().
          vis_dir: Directory where the visualizations will be saved.
        """

        tf.logging.info('Visualization for: {}'.format(
            samples[common.IMAGE_PATH][0].decode('utf8')))

        # Size of a visualization grid tile.
        tile_size = (300, 225)

        # Extension of the saved visualizations ('jpg', 'png', etc.).
        vis_ext = 'jpg'

        # Font settings.
        font_size = 10
        font_color = (0.8, 0.8, 0.8)

        # Intrinsics.
        K = samples[common.K][0]
        output_K = K * output_scale
        output_K[2, 2] = 1.0

        # Tiles for the grid visualization.
        tiles = []

        # Size of the output fields.
        output_size = \
            int(output_scale * crop_size[0]), int(output_scale * crop_size[1])

        # Prefix of the visualization names.
        vis_prefix = '{:06d}'.format(im_ind)

        # Input RGB image.
        rgb = np.squeeze(samples[common.IMAGE][0])
        vis_rgb = visualization.write_text_on_image(
            misc.resize_image_py(rgb, tile_size).astype(np.uint8),
            [{'name': '', 'val': 'input', 'fmt': ':s'}],
            size=font_size, color=font_color)
        tiles.append(vis_rgb)

        # Visualize the ground-truth poses.
        if self.vis_gt_poses:

            gt_poses = []
            for gt_id, obj_id in enumerate(samples[common.GT_OBJ_IDS][0]):
                q = samples[common.GT_OBJ_QUATS][0][gt_id]
                R = transform.quaternion_matrix(q)[:3, :3]
                t = samples[common.GT_OBJ_TRANS][0][gt_id].reshape((3, 1))
                gt_poses.append({'obj_id': obj_id, 'R': R, 't': t})

            vis_gt_poses = vis.visualize_object_poses(rgb, K, gt_poses, renderer)
            vis_gt_poses = visualization.write_text_on_image(
                misc.resize_image_py(vis_gt_poses, tile_size),
                [{'name': '', 'val': 'gt poses', 'fmt': ':s'}],
                size=font_size, color=font_color)
            tiles.append(vis_gt_poses)

        # Visualize the estimated poses.
        if self.vis_pred_poses:
            vis_pred_poses = vis.visualize_object_poses(rgb, K, pred_poses, renderer)
            vis_pred_poses = visualization.write_text_on_image(
                misc.resize_image_py(vis_pred_poses, tile_size),
                [{'name': '', 'val': 'pred poses', 'fmt': ':s'}],
                size=font_size, color=font_color)
            tiles.append(vis_pred_poses)

        # Ground-truth object labels.
        if self.vis_gt_obj_labels and common.GT_OBJ_LABEL in samples:
            obj_labels = np.squeeze(samples[common.GT_OBJ_LABEL][0])
            obj_labels = obj_labels[:crop_size[1], :crop_size[0]]
            obj_labels = vis.colorize_label_map(obj_labels)
            obj_labels = visualization.write_text_on_image(
                misc.resize_image_py(obj_labels.astype(np.uint8), tile_size),
                [{'name': '', 'val': 'gt obj labels', 'fmt': ':s'}],
                size=font_size, color=font_color)
            tiles.append(obj_labels)

        # Predicted object labels.
        if self.vis_pred_obj_labels:
            obj_labels = np.squeeze(predictions[common.PRED_OBJ_LABEL][0])
            obj_labels = obj_labels[:crop_size[1], :crop_size[0]]
            obj_labels = vis.colorize_label_map(obj_labels)
            obj_labels = visualization.write_text_on_image(
                misc.resize_image_py(obj_labels.astype(np.uint8), tile_size),
                [{'name': '', 'val': 'predicted obj labels', 'fmt': ':s'}],
                size=font_size, color=font_color)
            tiles.append(obj_labels)

        # Predicted object confidences.
        if self.vis_pred_obj_confs:
            num_obj_labels = predictions[common.PRED_OBJ_CONF].shape[-1]
            for obj_label in range(num_obj_labels):
                obj_confs = misc.resize_image_py(np.array(
                    predictions[common.PRED_OBJ_CONF][0, :, :, obj_label]), tile_size)
                obj_confs = (255.0 * obj_confs).astype(np.uint8)
                obj_confs = np.dstack([obj_confs, obj_confs, obj_confs])  # To RGB.
                obj_confs = visualization.write_text_on_image(
                    obj_confs, [{'name': 'cls', 'val': obj_label, 'fmt': ':d'}],
                    size=font_size, color=font_color)
                tiles.append(obj_confs)

        # Visualization of ground-truth fragment fields.
        if self.vis_gt_frag_fields and common.GT_OBJ_IDS in samples:
            vis.visualize_gt_frag(
                gt_obj_ids=samples[common.GT_OBJ_IDS][0],
                gt_obj_masks=samples[common.GT_OBJ_MASKS][0],
                gt_frag_labels=samples[common.GT_FRAG_LABEL][0],
                gt_frag_weights=samples[common.GT_FRAG_WEIGHT][0],
                gt_frag_coords=samples[common.GT_FRAG_LOC][0],
                output_size=output_size,
                model_store=model_store,
                vis_prefix=vis_prefix,
                vis_dir=vis_dir)

        # Visualization of predicted fragment fields.
        if self.vis_pred_frag_fields:
            vis.visualize_pred_frag(
                frag_confs=predictions[common.PRED_FRAG_CONF][0],
                frag_coords=predictions[common.PRED_FRAG_LOC][0],
                output_size=output_size,
                model_store=model_store,
                vis_prefix=vis_prefix,
                vis_dir=vis_dir,
                vis_ext=vis_ext)

        # Build and save a visualization grid.
        grid = vis.build_grid(tiles, tile_size)
        grid_vis_path = os.path.join(
            vis_dir, '{}_grid.{}'.format(vis_prefix, vis_ext))
        inout.save_im(grid_vis_path, grid)

    def save_correspondences(self,
            scene_id, im_id, im_ind, obj_id, image_path, K, obj_pred, pred_time,
            infer_name, obj_gt_poses, infer_dir):

        # Add meta information.
        txt = '# Corr format: u v x y z px_id frag_id conf conf_obj conf_frag\n'
        txt += '{}\n'.format(image_path)
        txt += '{} {} {} {}\n'.format(scene_id, im_id, obj_id, pred_time)

        # Add intrinsics.
        for i in range(3):
            txt += '{} {} {}\n'.format(K[i, 0], K[i, 1], K[i, 2])

        # Add ground-truth poses.
        txt += '{}\n'.format(len(obj_gt_poses))
        for pose in obj_gt_poses:
            for i in range(3):
                txt += '{} {} {} {}\n'.format(
                    pose['R'][i, 0], pose['R'][i, 1], pose['R'][i, 2], pose['t'][i, 0])

        # Sort the predicted correspondences by confidence.
        sort_inds = np.argsort(obj_pred['conf'])[::-1]
        px_id = obj_pred['px_id'][sort_inds]
        frag_id = obj_pred['frag_id'][sort_inds]
        coord_2d = obj_pred['coord_2d'][sort_inds]
        coord_3d = obj_pred['coord_3d'][sort_inds]
        conf = obj_pred['conf'][sort_inds]
        conf_obj = obj_pred['conf_obj'][sort_inds]
        conf_frag = obj_pred['conf_frag'][sort_inds]

        # Add the predicted correspondences.
        pred_corr_num = len(coord_2d)
        txt += '{}\n'.format(pred_corr_num)
        for i in range(pred_corr_num):
            txt += '{} {} {} {} {} {} {} {} {} {}\n'.format(
                coord_2d[i, 0], coord_2d[i, 1],
                coord_3d[i, 0], coord_3d[i, 1], coord_3d[i, 2],
                px_id[i], frag_id[i], conf[i], conf_obj[i], conf_frag[i])

        # Save the correspondences into a file.
        corr_suffix = infer_name
        if corr_suffix is None:
            corr_suffix = ''
        else:
            corr_suffix = '_' + corr_suffix

        corr_path = os.path.join(
            infer_dir, 'corr{}'.format(corr_suffix),
            '{:06d}_corr_{:02d}.txt'.format(im_ind, obj_id))
        tf.gfile.MakeDirs(os.path.dirname(corr_path))
        with open(corr_path, 'w') as f:
            f.write(txt)


    def process_image(self,
            sess, samples, predictions, im_ind, crop_size, output_scale, model_store,
            renderer, task_type, infer_name, infer_dir, vis_dir):
        """Estimates object poses from one image.

        Args:
          sess: TensorFlow session.
          samples: Dictionary with input data.
          predictions: Dictionary with predictions.
          im_ind: Index of the current image.
          crop_size: Image crop size (width, height).
          output_scale: Scale of the model output w.r.t. the input (output / input).
          model_store: Store for 3D object models of class ObjectModelStore.
          renderer: Renderer of class bop_renderer.Renderer().
          task_type: 6D object pose estimation task (common.LOCALIZATION or
            common.DETECTION).
          infer_name: Name of the current inference.
          infer_dir: Folder for inference results.
          vis_dir: Folder for visualizations.
        """
        # Dictionary for run times.
        run_times = {}

        # Prediction.
        time_start = time.time()
        (samples, predictions) = sess.run([samples, predictions])
        run_times['prediction'] = time.time() - time_start

        # Scene and image ID's.
        scene_id = samples[common.SCENE_ID][0]
        im_id = samples[common.IM_ID][0]

        # Intrinsic parameters.
        K = samples[common.K][0]

        if task_type == common.LOCALIZATION:
            gt_poses = []
            gt_obj_ids = samples[common.GT_OBJ_IDS][0]
            for gt_id in range(len(gt_obj_ids)):
                R = transform.quaternion_matrix(
                    samples[common.GT_OBJ_QUATS][0][gt_id])[:3, :3]
                t = samples[common.GT_OBJ_TRANS][0][gt_id].reshape((3, 1))
                gt_poses.append({'obj_id': gt_obj_ids[gt_id], 'R': R, 't': t})
        else:
            gt_poses = None

        # Establish many-to-many 2D-3D correspondences.
        time_start = time.time()
        corr = corresp.establish_many_to_many(
            obj_confs=predictions[common.PRED_OBJ_CONF][0],
            frag_confs=predictions[common.PRED_FRAG_CONF][0],
            frag_coords=predictions[common.PRED_FRAG_LOC][0],
            gt_obj_ids=[x['obj_id'] for x in gt_poses],
            model_store=model_store,
            output_scale=output_scale,
            min_obj_conf=self.corr_min_obj_conf,
            min_frag_rel_conf=self.corr_min_frag_rel_conf,
            project_to_surface=self.project_to_surface,
            only_annotated_objs=(task_type == common.LOCALIZATION))
        run_times['establish_corr'] = time.time() - time_start

        # PnP-RANSAC to estimate 6D object poses from the correspondences.
        time_start = time.time()
        poses = []
        for obj_id, obj_corr in corr.items():
            tf.logging.info(
              'Image path: {}, obj: {}'.format(samples[common.IMAGE_PATH][0], obj_id))

            # Number of established correspondences.
            num_corrs = obj_corr['coord_2d'].shape[0]

            # Skip the fitting if there are too few correspondences.
            min_required_corrs = 6
            if num_corrs < min_required_corrs:
                continue

            # The correspondences need to be sorted for PROSAC.
            if self.use_prosac:
                sorted_inds = np.argsort(obj_corr['conf'])[::-1]
                for key in obj_corr.keys():
                    obj_corr[key] = obj_corr[key][sorted_inds]

            # Select correspondences with the highest confidence.
            if self.max_correspondences is not None \
                    and num_corrs > self.max_correspondences:
                # Sort the correspondences only if they have not been sorted for PROSAC.
                if self.use_prosac:
                    keep_inds = np.arange(num_corrs)
                else:
                    keep_inds = np.argsort(obj_corr['conf'])[::-1]
                keep_inds = keep_inds[:self.max_correspondences]
                for key in obj_corr.keys():
                    obj_corr[key] = obj_corr[key][keep_inds]

            # Save the established correspondences (for analysis).
            if self.save_corresp:
                obj_gt_poses = []
                if gt_poses is not None:
                    obj_gt_poses = [x for x in gt_poses if x['obj_id'] == obj_id]
                pred_time = float(np.sum(list(run_times.values())))
                image_path = samples[common.IMAGE_PATH][0].decode('utf-8')
                self.save_correspondences(
                    scene_id, im_id, im_ind, obj_id, image_path, K, obj_corr, pred_time,
                    infer_name, obj_gt_poses, infer_dir)

            # Make sure the coordinates are saved continuously in memory.
            coord_2d = np.ascontiguousarray(obj_corr['coord_2d'].astype(np.float64))
            coord_3d = np.ascontiguousarray(obj_corr['coord_3d'].astype(np.float64))

            if self.fitting_method == common.PROGRESSIVE_X:
                # If num_instances == 1, then only GC-RANSAC is applied. If > 1, then
                # Progressive-X is applied and up to num_instances poses are returned.
                # If num_instances == -1, then Progressive-X is applied and all found
                # poses are returned.
                if task_type == common.LOCALIZATION:
                    num_instances = len([x for x in gt_poses if x['obj_id'] == obj_id])
                else:
                    num_instances = -1

                if self.max_instances_to_fit is not None:
                    num_instances = min(num_instances, self.max_instances_to_fit)

                pose_ests, inlier_indices, pose_qualities = pyprogressivex.find6DPoses(
                    x1y1=coord_2d,
                    x2y2z2=coord_3d,
                    K=K,
                    threshold=self.inlier_thresh,
                    neighborhood_ball_radius=self.neighbour_max_dist,
                    spatial_coherence_weight=self.spatial_coherence_weight,
                    scaling_from_millimeters=self.scaling_from_millimeters,
                    max_tanimoto_similarity=self.max_tanimoto_similarity,
                    max_iters=self.max_fitting_iterations,
                    conf=self.required_progx_confidence,
                    proposal_engine_conf=self.required_ransac_confidence,
                    min_coverage=self.min_hypothesis_quality,
                    min_triangle_area=self.min_triangle_area,
                    min_point_number=6,
                    max_model_number=num_instances,
                    max_model_number_for_optimization=self.max_model_number_for_pearl,
                    use_prosac=self.use_prosac,
                    log=False)

                pose_est_success = pose_ests is not None
                if pose_est_success:
                    for i in range(int(pose_ests.shape[0] / 3)):
                        j = i * 3
                        R_est = pose_ests[j:(j + 3), :3]
                        t_est = pose_ests[j:(j + 3), 3].reshape((3, 1))
                        poses.append({
                            'scene_id': scene_id,
                            'im_id': im_id,
                            'obj_id': obj_id,
                            'R': R_est,
                            't': t_est,
                            'score': pose_qualities[i],
                        })

            elif self.fitting_method == common.OPENCV_RANSAC:
                # This integration of OpenCV-RANSAC can estimate pose of only one object
                # instance. Note that in Table 3 of the EPOS CVPR'20 paper, the scores
                # for OpenCV-RANSAC were obtained with integrating cv2.solvePnPRansac
                # in the Progressive-X scheme (as the other methods in that table).
                pose_est_success, r_est, t_est, inliers = cv2.solvePnPRansac(
                    objectPoints=coord_3d,
                    imagePoints=coord_2d,
                    cameraMatrix=K,
                    distCoeffs=None,
                    iterationsCount=self.max_fitting_iterations,
                    reprojectionError=self.inlier_thresh,
                    confidence=0.99,  # FLAGS.required_ransac_confidence
                    flags=cv2.SOLVEPNP_EPNP)

                if pose_est_success:
                    poses.append({
                        'scene_id': scene_id,
                        'im_id': im_id,
                        'obj_id': obj_id,
                        'R': cv2.Rodrigues(r_est)[0],
                        't': t_est,
                        'score': 0.0,  # TODO: Define the score.
                    })

            else:
                raise ValueError(
                    'Unknown pose fitting method ({}).'.format(self.fitting_method))

        run_times['fitting'] = time.time() - time_start
        run_times['total'] = np.sum(list(run_times.values()))

        # Add the total time to each pose.
        for pose in poses:
            pose['time'] = run_times['total']

        # Visualization.
        if self.vis:
            self.visualize(
                samples=samples,
                predictions=predictions,
                pred_poses=poses,
                im_ind=im_ind,
                crop_size=crop_size,
                output_scale=output_scale,
                model_store=model_store,
                renderer=renderer,
                vis_dir=vis_dir)

        return poses, run_times

# Class for service related data handling
class cameraInput:
    def __init__(self):
        self.enableInputCallbacks = False
        self.rgb = None
        self.depth = None
        self.K = None
        self.rgbImageTimestamp = 0
        self.objectId=0
        self.posesPredicted=None
        self.img1Count=-1
        self.img2Count=-1
        self.pub = rospy.Publisher('/ICCS/ObjectDetectionAndLocalization/ObjectPose', ObjectPose, queue_size=10)
        self.ImageReady = False

    def callbackRGBImage(self, data):
        if self.enableInputCallbacks == True:
            try:
                self.rgb = CvBridge().imgmsg_to_cv2(data, desired_encoding="bgr8")
                self.rgbImageTimestamp = data.header.stamp
                im_rgb = cv2.cvtColor(self.rgb, cv2.COLOR_BGR2RGB)

                # Clean up image folders
                files1 = glob.glob('/home/lele/Codes/epos/datasets/carObj1/test_primesense/000001/rgb/*.png', recursive=True)
                files2 = glob.glob('/home/lele/Codes/epos/datasets/carObj1/test_primesense/000002/rgb/*.png', recursive=True)
                for f in files1:
                    try:
                        os.remove(f)
                    except OSError as e:
                        print("Error: %s : %s" % (f, e.strerror))
                for f in files2:
                    try:
                        os.remove(f)
                    except OSError as e:
                        print("Error: %s : %s" % (f, e.strerror))

                # capture and save images
                if self.objectId == 1:
                    self.img1Count += 1
                    self.ImageReady = False
                    imageio.imwrite('/home/lele/Codes/epos/datasets/carObj1/test_primesense/000001/rgb/000000.png', im_rgb)
                    while not self.ImageReady:
                        imgTest1 = cv2.imread(
                            '/home/lele/Codes/epos/datasets/carObj1/test_primesense/000001/rgb/000000.png')
                        if imgTest1.size != 0:
                            self.ImageReady = True
                elif self.objectId == 2:
                    self.img2Count += 1
                    imageio.imwrite('/home/lele/Codes/epos/datasets/carObj1/test_primesense/000002/rgb/000000.png',  im_rgb)
                    while not self.ImageReady:
                        imgTest2 = cv2.imread(
                            '/home/lele/Codes/epos/datasets/carObj1/test_primesense/000002/rgb/000000.png')
                        if imgTest2.size != 0:
                            self.ImageReady = True

            except CvBridgeError as e:
                print(e)

    def callbackDepthImage(self, data):
        if self.enableInputCallbacks == True:
            try:
                self.depth = CvBridge().imgmsg_to_cv2(data, desired_encoding="passthrough")
                if self.objectId == 1:
                    path = '/home/lele/Codes/epos/datasets/carObj1/test_primesense/000001/depth/000000.png'
                elif self.objectId == 2:
                    path = '/home/lele/Codes/epos/datasets/carObj1/test_primesense/000002/depth/000000.png'
                im_uint16 = np.round(self.depth).astype(np.uint16)
                # PyPNG library can save 16-bit PNG and is faster than imageio.imwrite().
                w_depth = png.Writer(self.depth.shape[1], self.depth.shape[0], greyscale=True, bitdepth=16)
                with open(path, 'wb') as f:
                    w_depth.write(f, np.reshape(im_uint16, (-1, self.depth.shape[1])))
            except CvBridgeError as e:
                print(e)

    def callbackCameraInfo(self, data):
        if self.enableInputCallbacks == True:
            #rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.K)
            self.K = data.K

    def get_pose(self, req):
        # distinguish requested object
        if req.objID ==1:
            self.objectId = req.objID
        elif req.objID == 13:
            self.objectId = 2
        else:
            raise ValueError("Invalid object ID")

        # Flag for capturing image when service is called
        self.enableInputCallbacks = True
        pose = ObjectPose()

        while self.rgb is None or self.depth is None or self.K is None or not self.ImageReady:
            pass

        # Flag for verifying image writing is complete
        self.enableInputCallbacks = False

        # Initializations
        rgb = self.rgb
        depth = self.depth
        K = self.K
        timeImage=self.rgbImageTimestamp

        sc = 0.0
        pos = [0.0, 0.0, 0.0]
        orie = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        uL = [0, 0]
        lR = [0, 0]

        # Create example lists and tf records (input for epos)
        exampleListCreator = createExampleListCls()
        exampleListCreator.createExampleList()
        tfRecordsCreator = createTfRecordsCls()
        tfRecordsCreator.runCreateTfRecords()

        # Run epos prediction
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        infer = eposInfer()
        self.posesPredicted = infer.predictPose()

        # Assign predicted pose
        first_im_poses_num = len(self.posesPredicted)
        for i in range(first_im_poses_num):
           if self.objectId == self.posesPredicted[i]['obj_id']:
               sc = self.posesPredicted[i]['score']
               pos = self.posesPredicted[i]['t']
               orie = self.posesPredicted[i]['R'].flatten('C')
               #uL = [0, 0]
               #lR = [0, 0]

        # create assign values to the results message
        pose.timestamp = rospy.Time.now()  # .get_rostime()
        pose.timestampImage = timeImage
        pose.score = sc
        pose.objID = req.objID
        pose.position = pos
        pose.orientation = orie
        pose.uLCornerBB = uL
        pose.lRCornerBB = lR

        # clean up inputs to get prepared for the next call
        self.rgb = None
        self.depth = None
        self.K = None

        # cleanup folders
        files3 = glob.glob('/home/lele/Codes/epos/store/tf_data/example_lists/*.txt', recursive=True)
        #files4 = glob.glob('/home/lele/Codes/epos/store/tf_data/*.tfrecord', recursive=True)

        for f in files3:
           try:
               os.remove(f)
           except OSError as e:
               print("Error: %s : %s" % (f, e.strerror))
        # for f in files4:
        #    try:
        #        os.remove(f)
        #    except OSError as e:
        #        print("Error: %s : %s" % (f, e.strerror))

        self.pub.publish(pose)

        # clean up cuda
        gc.collect() #https://stackoverflow.com/questions/39758094/clearing-tensorflow-gpu-memory-after-model-execution

        return ObjectPoseServiceResponse(pose)

def publisherHealth():
    pub = rospy.Publisher('/ICCS/ObjectDetectionAndLocalization/SystemHealth', SystemHealth, queue_size=10)
    #rospy.init_node('healthPub', anonymous=True)  # define the ros node - publish node
    rate = rospy.Rate(10)  # 10hz frequency at which to publish

    if not rospy.is_shutdown():
        msg = SystemHealth()
        msg.timestamp = rospy.Time.now()  # .get_rostime()
        msg.status = "OK"

        rospy.loginfo(msg)  # to print on the terminal
        pub.publish(msg)  # publish
        rate.sleep()

def sendHealthMsg():
  threading.Timer(5.0, sendHealthMsg).start()
  #status = "OK"
  publisherHealth()

def runODL():
    camInput = cameraInput()
    rospy.init_node('ObjPose', anonymous=True)
    sendHealthMsg()
    rospy.Subscriber('/camera/color/image_raw', Image, camInput.callbackRGBImage)
    rospy.Subscriber('/camera/color/camera_info', CameraInfo, camInput.callbackCameraInfo)
    rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, camInput.callbackDepthImage)
    s = rospy.Service('object_pose', ObjectPoseService, camInput.get_pose)
    rospy.spin()


if __name__ == "__main__":
    runODL()

