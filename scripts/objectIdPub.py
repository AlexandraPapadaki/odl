#!/usr/bin/env python

import rospy
from beginner_tutorials.msg import ObjectId

def publisherObjectId():
    pub = rospy.Publisher('/ICCS/ObjectDetectionAndLocalization/ObjectId', ObjectId, queue_size=10) 
    rospy.init_node('objectIdPub', anonymous=True) # define the ros node - publish node
    rate = rospy.Rate(10) # 10hz frequency at which to publish
    
    if not rospy.is_shutdown():
        msg = ObjectId()
        msg.timestamp = rospy.Time.now()#.get_rostime()
        msg.objID = ID

        rospy.loginfo(msg) # to print on the terminal
        pub.publish(msg) # publish
        rate.sleep()

if __name__ == '__main__':
    try:        
        ID = 13
        
        publisherObjectId()
    except rospy.ROSInterruptException:
        pass
