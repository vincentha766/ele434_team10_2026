#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =========================================================================
# 节点名称: coverage_navigation_node
# 适用场景: ROS 2 (Waffle), 4x4m场地外围探索任务
# 核心策略: 
#   1. 矩形包络线严格得分判定 (车身完全进入)
#   2. 顺时针拓扑 + 墙角强制右转逃逸 (防止死锁)
#   3. 动态注意力势场法 (防圆筒死锁)
#   4. 阿克曼转向约束 (绝对禁止原地打转)
# =========================================================================

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import math

class CoverageNavigationNode(Node):
    def __init__(self):
        super().__init__('coverage_navigation_node')
        
        # =====================================================================
        # [1] 可调参数配置 (未来可通过 ROS 2 Param 系统实时动态修改)
        # =====================================================================
        self.p_max_v = 0.25           # 最大线速度 (m/s)，Waffle 的安全极速
        self.p_max_w = 1.5            # 最大角速度 (rad/s)，控制转向的爆发力
        self.p_safe_dist = 0.35       # 避障预警距离 (m)，距离小于此值开始产生排斥力
        self.p_corner_weight = 1.0    # 墙角逃逸斥力权重，调大可加速逃离90度死角
        self.enable_repeat = False     # 是否循环跑圈 (True为无限循环跑，False为跑完一圈自动停车)
        
        # =====================================================================
        # [2] 内部固定物理安全参数 (与车体物理尺寸绑定，不建议随意修改)
        # =====================================================================
        self.BOX_LIMIT = 0.30         # 得分判定容差：0.5m(区块半宽) - 0.2m(车体半径) = 0.3m
        self.STOP_DIST = 0.22         # 物理防撞极限距离：比车身半径略大，突破此距离将强制触发底线保护
        self.MIN_FORWARD_SPEED = 0.06 # 最低前进速度约束：强制机器人在避障时也要保持向前滑行，禁止像坦克一样原地旋转

        # =====================================================================
        # [3] 场地地图与状态机初始化
        # =====================================================================
        # 定义 12 个外围得分区块的中心坐标 (严格遵循顺时针拓扑)
        # 索引 0 代表 1 号区域 (右上角)
        self.waypoints = [
             (1.5,  0.5), (1.5, -0.5),  (1.5, -1.5), (0.5, -1.5), # 索引 1,2,3,4 (场地右侧下行)
             (-0.5, -1.5), (-1.5, -1.5), (-1.5, -0.5), (-1.5, 0.5), # 索引 5,6,7,8 (场地底部左行)
             (-1.5,  1.5), (-0.5,  1.5),  (0.5,  1.5),(1.5,  1.5)   # 索引 9,10,11,0(场地顶部右行)
        ]
        self.num_waypoints = len(self.waypoints)
        self.current_idx = 0  # 状态机：当前正在前往的航点索引 (初始化为 0，即右上角 1 号点)
        
        # 传感器就绪标志位 (防止在没收到数据时就执行控制计算引发报错)
        self.odom_ready = False
        self.lidar_ready = False
        self.initialized_start_point = False # 记录是否已经发送了起始提示
        
        # 机器人当前实时状态缓存
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.lidar_ranges = []

        # =====================================================================
        # [4] ROS 2 通信接口 (Publishers / Subscribers)
        # =====================================================================
        # 创建发布者：发布底盘控制指令 (话题: /cmd_vel)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # 创建订阅者：订阅里程计数据 (获取实时坐标与姿态)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # 创建订阅者：订阅激光雷达数据 (获取全向避障距离)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        
        # 创建核心控制循环定时器：20Hz (0.05秒执行一次 control_loop)
        self.timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info("覆盖导航节点已启动，正在等待 Odom 与 Lidar 数据...")

    # =====================================================================
    # 辅助数学计算函数
    # =====================================================================
    def wrap_to_pi(self, angle):
        """
        角度标准化函数：将任意偏航误差收敛到 [-pi, pi] 范围内。
        这能保证机器人总是选择最短的旋转方向 (比如需要转 270 度时，算法会将其换算为反向转 -90 度)
        """
        return (angle + math.pi) % (2 * math.pi) - math.pi

    # =====================================================================
    # 传感器回调函数 (Callbacks)
    # =====================================================================
    def odom_callback(self, msg):
        """
        里程计回调函数：频率极高，持续更新机器人的 X/Y 坐标与朝向。
        """
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        
        # 将三维空间复杂的四元数 (Quaternion) 转换为简单的二维平面偏航角 (Yaw/Euler Angle)
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.odom_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        self.odom_ready = True # 标记里程计数据已就绪

    def scan_callback(self, msg):
        """
        激光雷达回调函数：提取并清洗 360 度距离数据。
        """
        cleaned_ranges = []
        for r in msg.ranges:
            # 过滤掉物理雷达常见的噪点：无穷大 (inf)、无效数据 (nan) 以及紧贴传感器的外壳反射噪点 (< 5cm)
            if math.isnan(r) or math.isinf(r) or r < 0.05:
                cleaned_ranges.append(3.5) # 用最大安全距离 (3.5m) 替代，代表该方向非常空旷
            else:
                cleaned_ranges.append(r)
        
        self.lidar_ranges = cleaned_ranges
        self.lidar_ready = True # 标记雷达数据已就绪

    # =====================================================================
    # 核心大脑：控制循环 (20Hz 定时触发)
    # =====================================================================
    def control_loop(self):
        # 保护机制：如果雷达或里程计还没连上，什么都不做，直接跳过
        if not self.odom_ready or not self.lidar_ready:
            return

        # 启动时的首次提示
        if not self.initialized_start_point:
            self.current_idx = 0 # 严格从数组第一个元素 (1号区域) 开始
            self.initialized_start_point = True
            self.get_logger().info(f"传感器就绪！前往 1 号区域，中心坐标: {self.waypoints[self.current_idx]}")

        # 如果已经跑完了所有点且没有开启循环，执行停车逻辑
        if self.current_idx >= self.num_waypoints:
            self.stop_robot()
            return

        # ---------------------------------------------------------------------
        # [步骤 A] 状态机：计算偏差与【完全进入】得分判定
        # ---------------------------------------------------------------------
        target_x, target_y = self.waypoints[self.current_idx]
        dx = target_x - self.odom_x
        dy = target_y - self.odom_y
        
        # 判定条件：X 和 Y 轴的坐标差必须同时小于 0.30m，证明半径 0.2m 的车体完全位于 1x1m 的区块内
        if abs(dx) < self.BOX_LIMIT and abs(dy) < self.BOX_LIMIT:
            self.current_idx += 1 # 打卡成功，切换索引
            self.get_logger().info(f"完全进入区域 {self.current_idx} (打卡成功)，切换至下一区域。")
            
            # 越界检查 (是否跑完了第 12 个点)
            if self.current_idx >= self.num_waypoints:
                if self.enable_repeat:
                    self.current_idx = 0 # 重新回到起点，继续跑圈
                    self.get_logger().info("🔁 完成一圈拓扑，重新开始循环！")
                else:
                    self.get_logger().info("🏁 任务全部完成！申请停车。")
                    self.stop_robot()
                    return
            
            # 索引更新后，立即刷新目标坐标和偏差，供后面的控制算法使用
            target_x, target_y = self.waypoints[self.current_idx]
            dx = target_x - self.odom_x
            dy = target_y - self.odom_y

        # ---------------------------------------------------------------------
        # [步骤 B] 雷达高精度 8 扇区划分 (以车头为 0 度)
        # ---------------------------------------------------------------------
        num_scans = len(self.lidar_ranges)
        if num_scans >= 360:
            # 拼接 337~360度 和 0~22度，组合成正前方 45度角视场
            r_f  = self.lidar_ranges[337:360] + self.lidar_ranges[0:22] 
            r_fl = self.lidar_ranges[22:67]   # 左前方
            r_l  = self.lidar_ranges[67:112]  # 正左方
            r_r  = self.lidar_ranges[247:292] # 正右方
            r_fr = self.lidar_ranges[292:337] # 右前方
        else:
            # 针对雷达数据异常的兜底防崩溃处理
            r_f = r_fl = r_l = r_r = r_fr = self.lidar_ranges

        # 提取各个关键扇区距离我们最近的障碍物距离
        min_f  = min(r_f)
        min_fl = min(r_fl)
        min_fr = min(r_fr)
        min_l  = min(r_l)
        min_r  = min(r_r)

        # ---------------------------------------------------------------------
        # [步骤 C] 核心势场计算：目标引力 vs 障碍物斥力 (顺时针增强版)
        # ---------------------------------------------------------------------
        # 计算理论上指向目标点的最优偏航角，及角度误差
        target_angle = math.atan2(dy, dx)
        angle_error = self.wrap_to_pi(target_angle - self.odom_yaw)
        
        # 1. 动态引力削弱 (注意力转移机制)
        # 获取全局最危险的距离。如果靠墙极近，机器人会逐渐把注意力(权重)从目标点转移到避障上。
        min_scan = min(min_f, min_fl, min_fr, min_l, min_r)
        if min_scan < self.p_safe_dist:
            # 线性衰减：距离越近，权重越小；逼近 STOP_DIST 时权重归 0，完全放弃寻路
            track_weight = max(0.0, (min_scan - self.STOP_DIST) / (self.p_safe_dist - self.STOP_DIST))
        else:
            track_weight = 1.0 # 环境空旷，全心全意前往目标
            
        # 基础引力角速度分量
        omega_track = angle_error * 1.6 * track_weight
        
        # 2. 障碍物斥力计算
        repel_omega = 0.0
        
        # 【死胡同/墙角陷阱检测】：如果左前和右前同时存在预警，说明进入了 V 型墙角夹击
        is_in_corner = (min_fl < self.p_safe_dist) and (min_fr < self.p_safe_dist)
        
        if is_in_corner:
            # 判断当前是否处于赛道外围的四个主转向角 (1, 4, 7, 10 号区域，对应列表索引 0, 3, 6, 9)
            is_major_corner = self.current_idx in [0, 3, 6, 9]
            
            if is_major_corner:
                # 【重磅逻辑：顺时针大局观】
                # 到了场地的四个边角，为了顺时针贴边，无视任何雷达数据，无脑输出极其强烈的向右斥力！
                repel_omega = -self.p_corner_weight * 2.2
            else:
                # 处于非主墙角区域 (例如被场地中间圆柱和挡板夹住)：
                # 判断两边哪边空旷就往哪边跑。特意为右侧加了 0.15m 的“优先偏置”，鼓励小车优先右转逃跑。
                if (min_fl + min_l) > (min_fr + min_r) + 0.15:
                    repel_omega = self.p_corner_weight * 1.8  # 左边大很多，向左逃
                else:
                    repel_omega = -self.p_corner_weight * 1.8 # 否则默认向右逃
        else:
            # 【常规滑行避障】：根据距离产生与距离成反比的反向推力
            if min_fl < self.p_safe_dist: repel_omega -= (self.p_safe_dist - min_fl) * 6.0 # 左前有墙，向右推
            if min_fr < self.p_safe_dist: repel_omega += (self.p_safe_dist - min_fr) * 6.0 # 右前有墙，向左推
            # 纯侧面防擦碰微调 (限位 0.28m 内触发)
            if min_l < 0.28: repel_omega -= (0.28 - min_l) * 4.0
            if min_r < 0.28: repel_omega += (0.28 - min_r) * 4.0

        # 【正前方圆筒死锁破局】
        # 如果正前方被圆筒挡住，不进行特殊处理的话车会左右摇摆。
        if min_f < self.p_safe_dist and not is_in_corner:
            # 依然带有右转基因：除非左侧明显开阔 (+0.15m)，否则默认提供向右的破局偏置
            bias_dir = 1.5 if (min_fl + min_l) > (min_fr + min_r) + 0.15 else -1.5
            repel_omega += bias_dir * (self.p_safe_dist - min_f)

        # ---------------------------------------------------------------------
        # [步骤 D] 动力学指令合成与阿克曼强制约束 (禁止原地旋转)
        # ---------------------------------------------------------------------
        # 1. 最终合成角速度 (引力 + 斥力)，并进行硬限幅防失控
        final_w = omega_track + repel_omega
        angular_vel = max(min(final_w, self.p_max_w), -self.p_max_w)

        # 2. 线速度动态削减算法
        # 削减基础：需要转的弯越大，直线速度压得越低，防止被离心力甩出边界
        speed_factor = 1.0 - (abs(angle_error) / math.pi)
        
        # 避障降速：前方越危险，强制线速度上限越低
        if min_f < self.p_safe_dist:
            obs_limit = max(0.05, (min_f - self.STOP_DIST) / (self.p_safe_dist - self.STOP_DIST))
            speed_factor = min(speed_factor, obs_limit)
        
        # 计算初始期望线速度
        linear_vel = self.p_max_v * max(speed_factor, 0.2)
        
        # 【关键保底约束：绝对禁止原地旋转】
        # 无论转向角多大，无论前方多危险，强制线速度不能低于 0.06 m/s。
        # 在 v ≠ 0, w ≠ 0 的情况下，机器人的物理轨迹必定是一个圆弧绕行，完美规避原地打转违规。
        if linear_vel < self.MIN_FORWARD_SPEED:
            linear_vel = self.MIN_FORWARD_SPEED
            
        # 极限物理防撞：即将撞墙时剥夺算法主控权，只允许维持 0.06m/s 的蠕行速度进行急转弯
        if min_f <= self.STOP_DIST:
            linear_vel = self.MIN_FORWARD_SPEED

        # ---------------------------------------------------------------------
        # [步骤 E] 组装并发布 ROS 2 消息
        # ---------------------------------------------------------------------
        cmd = Twist()
        cmd.linear.x = float(linear_vel)
        cmd.angular.z = float(angular_vel)
        self.cmd_pub.publish(cmd)

    def stop_robot(self):
        """
        急停函数：发送零速度指令，让电机彻底停止。
        """
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)
        self.get_logger().info("发送停车指令！")

# =====================================================================
# 节点入口点 (Entry Point)
# =====================================================================
def main(args=None):
    # 初始化 rclpy 上下文
    rclpy.init(args=args)
    # 实例化我们的导航节点
    node = CoverageNavigationNode()
    
    try:
        # 进入事件循环，阻塞当前线程，不断处理定时器和订阅数据的回调
        rclpy.spin(node)
    except KeyboardInterrupt:
        # 捕捉用户按下的 Ctrl+C
        node.get_logger().info("检测到键盘中断，正在停止机器人...")
    finally:
        # 无论如何，退出前确保发送了刹车指令，清理节点资源
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()