import argparse

from KeyFrameDetector.key_frame_detector import keyframeDetection

def main():
    parser = argparse.ArgumentParser()

    # parser.add_argument('-s','--source', default='/data/yisi/mywork/omnimotion/video-keyframe-detector-master/videos/acrobacia.mp4',
    #                     help='source file', required=True)
    # parser.add_argument('-d', '--dest', default='/data/yisi/mywork/omnimotion/video-keyframe-detector-master/output',
    #                     help='destination folder', required=True)
    # parser.add_argument('-t','--Thres', help='Threshold of the image difference', default=0.3)

    parser.add_argument('-s', default='/data/yisi/mywork/ominimotion_plus/video-keyframe-detector-master/videos/bear.mp4',
                        help='source file')
    parser.add_argument('-d',  default='/data/yisi/mywork/ominimotion_plus/video-keyframe-detector-master/keyframe_bear',
                        help='destination folder')
    parser.add_argument('-t', help='Threshold of the image difference', default=0.3)

    args = parser.parse_args()


    # keyframeDetection(args.source, args.dest, float(args.Thres))
    keyframeDetection(args.s, args.d, float(args.t), verbose=True)

if __name__ == '__main__':
    main()
