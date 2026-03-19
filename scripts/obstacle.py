#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped

class Node(Node):
    
    def __init__(self, name):
        super().__init__(name)                                    # ROS2节点父类初始化
        self.sub = self.create_subscription(
            LaserScan, 
            "/scan", 
            self.LidarCallback,
            rclpy.qos.qos_profile_sensor_data)         # 创建订阅者对象（消息类型、话题名、订阅者回调函数、队列长度）
        
        self.pub = self.create_publisher(TwistStamped, 
                                         "/cmd_vel", 
                                         10)
        self.vel_cmd = TwistStamped()

    #小车行为        
    def turn(self, turn):
        self.vel_cmd.twist.angular.z = turn
        self.vel_cmd.twist.linear.x = 0.25
        self.pub.publish(self.vel_cmd)    
    
    def move(self, move):
        self.vel_cmd.twist.linear.x = move
        self.pub.publish(self.vel_cmd)
        
   
    def juggment(self,dist):#判断函数   
        #获取采样长度
        lon = len(dist)
        #用于归一化角度的系数
        i = lon/360 
        
        
        
        
                  
                                                 
    def LidarCallback(self, msg):# 创建回调函数，执行收到话题消息后对数据的处理
        dist = msg.ranges                   
        self.juggment(dist)
          
        
def main():
    rclpy.init()# 初始化节点    
    node = Node('laser_touch')
    rclpy.spin(node)# 让节点保持运行
    rclpy.destroy_node()
    rclpy.shutdown()
    exit()

if __name__ == '__main__':
    main()