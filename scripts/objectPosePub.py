#!/usr/bin/env python

import rospy
from odl.msg import ObjectPose

def publisher():
    pub = rospy.Publisher('/ICCS/ObjectDetectionAndLocalization/ObjectPose', ObjectPose, queue_size=10) 
    rospy.init_node('objectPosePub', anonymous=True) # define the ros node - publish node
    rate = rospy.Rate(10) # 10hz frequency at which to publish
    
    if not rospy.is_shutdown():
        msg = ObjectPose()
        msg.timestamp = rospy.Time.now()#.get_rostime()
        msg.score = sc
        msg.objID = ID
        msg.position = pos
        msg.orientation = orie
        msg.uLCornerBB = uL
        msg.lRCornerBB = lR

        rospy.loginfo(msg) # to print on the terminal
        pub.publish(msg) # publish
        rate.sleep()

if __name__ == '__main__':
    try:        
        sc = 2.2
        ID = 1
        pos = [0.0, 0.1, 0.2]
        orie = [2.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
        uL = [200, 201]
        lR= [900, 301]
        
        publisher()
    except rospy.ROSInterruptException:
        pass
