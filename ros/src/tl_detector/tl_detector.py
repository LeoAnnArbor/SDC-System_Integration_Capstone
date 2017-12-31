#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        # rospy.spin()
        self.loop()

    def loop(self):
        
        rate = rospy.Rate(5) #Hz

        while not rospy.is_shutdown():

            if self.pose is not None and self.waypoints is not None and self.has_image:
                
                light_wp, state = self.process_traffic_lights()

                '''
                Publish upcoming red lights at camera frequency.
                Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
                of times till we start using it. Otherwise the previous stable state is
                used.
                '''
                if self.state != state:
                    self.state_count = 0
                    self.state = state
                elif self.state_count >= STATE_COUNT_THRESHOLD:
                    self.last_state = self.state
                    light_wp = light_wp if state == TrafficLight.RED else -1
                    self.last_wp = light_wp
                    self.upcoming_red_light_pub.publish(Int32(light_wp))
                else:
                    self.upcoming_red_light_pub.publish(Int32(self.last_wp))
                self.state_count += 1
            
            rate.sleep()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        if not self.has_image:
            rospy.loginfo("Received image from simulator")
            
        self.has_image = True
        self.camera_image = msg
        # light_wp, state = self.process_traffic_lights()

        # '''
        # Publish upcoming red lights at camera frequency.
        # Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        # of times till we start using it. Otherwise the previous stable state is
        # used.
        # '''
        # if self.state != state:
        #     self.state_count = 0
        #     self.state = state
        # elif self.state_count >= STATE_COUNT_THRESHOLD:
        #     self.last_state = self.state
        #     light_wp = light_wp if state == TrafficLight.RED else -1
        #     self.last_wp = light_wp
        #     self.upcoming_red_light_pub.publish(Int32(light_wp))
        # else:
        #     self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        # self.state_count += 1

    def get_closest_index(self, pose, pose_list):
        """Identifies the closest lights to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to
            post_list: a list of positions (of traffic lights)
        Returns:
            int: index of the closest waypoint in self.waypoints
        """
        
        min_dist = 1e100
        index = 0
        
        for idx, point in enumerate(pose_list):

            dist = math.hypot(point.pose.pose.position.x-pose.position.x, point.pose.pose.position.y-pose.position.y)

            if dist < min_dist:

                min_dist = dist
                index = idx

        return index


    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def generate_stop_line(self, x, y, z):
        
        light = TrafficLight()
        light.pose = PoseStamped()
        light.pose.pose.position.x = x
        light.pose.pose.position.y = y
        light.pose.pose.position.z = z
        
        return light

    def process_traffic_lights(self):
        
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        
        if(self.pose and self.waypoints):

            light_position = self.get_closest_index(self.pose.pose, self.lights)

            stop_lines = list()
            for stop_line_pos in stop_line_positions:
                stop_line =  self.generate_stop_line(stop_line_pos[0], stop_line_pos[1], 0.)
                stop_lines.append(stop_line)
            
            stop_line_wp = self.get_closest_index(stop_lines[light_position].pose.pose, self.waypoints.waypoints)
            
            state = self.get_light_state(self.lights[light_position])
            rospy.loginfo_throttle(2, "Light: " + str(state))
            return stop_line_wp, state

        #TODO find the closest visible traffic light (if one exists)

        if light:
            state = self.get_light_state(light)
            return light_wp, state
        # self.waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
