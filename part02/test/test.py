#! /usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
import time

def test():
	msg=Twist()
	pub=rospy.Publisher('/cmd_vel',Twist,queue_size=10)
	rospy.init_node('robot_talker',anonymous=True)
	while not rospy.is_shutdown():
		msg.angular.z=0
		msg.linear.x=-0.1
		#buff='my current time is %s" %rospy.get_time()
		pub.publish(msg)
		time.sleep(0.1)
		
if __name__=='__main__':
	test()
	
