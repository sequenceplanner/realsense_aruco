import cv2
import numpy as np
import pyrealsense2 as rs


pipeline = None
align = None
clipping_distance = None


def init():
    global pipeline
    global align
    global clipping_distance

    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth scale is {depth_scale}")

    clipping_distance_meter = 1
    clipping_distance = clipping_distance_meter / depth_scale

    align_to = rs.stream.color
    align = rs.align(align_to)


def get_frame():
    frames = pipeline.wait_for_frames()

    aligned_frames = align.process(frames)

    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        return False, None, None
        
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    grey_color = 153
    depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
    bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d < 0), grey_color, color_image)

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    images = np.hstack((bg_removed, depth_colormap))

    return True, depth_image, color_image, images


def stream_frames():
    while True:
        ret, depth_frame, color_frame, images = get_frame()

        pt_x, pt_y = pt = 400, 300


        cv2.circle(color_frame, pt, 4, (0, 0, 255))
        distance = depth_frame[pt_y, pt_x]
        print(distance)
        cv2.imshow("depth frame", depth_frame)
        cv2.imshow("color frame", color_frame)
        cv2.imshow("images", images)


        input_key = cv2.waitKey(1)
        if input_key == 27:
            break



def release():
    pipeline.stop()



def main():
    init()
    
    stream_frames()

    release()

if __name__ == "__main__":
    main()
