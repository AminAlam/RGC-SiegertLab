import numpy
from visexpman.engine.vision_experiment import experiment
from visexpman.engine.generic import utils
import cv2

class NaturalMovie(experiment.Stimulus):
    def configuration(self):
        self.MOVIE_PATH='C:\\Users\\maxone\\Downloads\\natural_movie.mp4'
        self.IMAGES_PATH='C:\\Users\\maxone\\Downloads\\\\test\\'
        self.WAIT=0.5#wait time in seconds at beginning and end of stimulus
        self.REPEATS=1
        self.BACKGROUND=0.0
        self.QUALITY_FACTOR=0.5#quality factor of the movie. 1 is the original quality
        self.FS=60#frequency which makes the time vector. should be < frame rate of the projector

#Do not edit below this!

    def make_image_sequence_from_movie(self):
        # Load movie as frames
        cap = cv2.VideoCapture(self.MOVIE_PATH)
        frames_list = []
        while(cap.isOpened()):
            ret, frame = cap.read()
            self.MOVIE_FRAME_RATE = cap.get(cv2.CAP_PROP_FPS)
            # get frame_rate of the movie
            if ret==True:
                frames_list.append(frame)
            else:
                break
        cap.release()
        self.IMAGES_PATH_LIST = []
        # save frames as images in self.IMAGES_PATH
        for i in range(len(frames_list)):
            img = frames_list[i]
            if self.QUALITY_FACTOR < 1:
                img = cv2.resize(img, (0,0), fx=self.QUALITY_FACTOR, fy=self.QUALITY_FACTOR)
            image_path = self.IMAGES_PATH + 'frame' + str(i) + '.jpg'
            self.IMAGES_PATH_LIST.append(image_path)
            cv2.imwrite(image_path, img)


    def prepare(self):
        self.make_image_sequence_from_movie()
        if self.FS > self.MOVIE_FRAME_RATE:
            self.FS = self.MOVIE_FRAME_RATE//10 * 10
 
    def run(self):
        self.show_fullscreen(color=self.BACKGROUND, duration=self.WAIT)
        for r in range(self.REPEATS):
            for img_path in self.IMAGES_PATH_LIST:
                self.block_start(('on',))
                self.show_image(img_path, duration=1/self.FS, stretch=1/self.QUALITY_FACTOR)
                self.block_end()
                if self.abort:
                    break
        self.show_fullscreen(color=self.BACKGROUND, duration=self.WAIT)
