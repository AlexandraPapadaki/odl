#!/usr/bin/env python

import rospy
from odl.msg import Health

def publisherHealth():
    pub = rospy.Publisher('/ICCS/ObjectDetectionAndLocalization/Health', Health, queue_size=10) 
    rospy.init_node('healthPub', anonymous=True) # define the ros node - publish node
    rate = rospy.Rate(10) # 10hz frequency at which to publish
    
    if not rospy.is_shutdown():
        msg = Health()
        msg.timestamp = rospy.Time.now()#.get_rostime()
        msg.status = status

        rospy.loginfo(msg) # to print on the terminal
        pub.publish(msg) # publish
        rate.sleep()

if __name__ == '__main__':
    try:        
        status = "OK"
        
        publisherHealth()
    except rospy.ROSInterruptException:
        pass
