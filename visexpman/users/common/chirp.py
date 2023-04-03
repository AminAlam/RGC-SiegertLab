import numpy
from visexpman.engine.vision_experiment import experiment
from visexpman.engine.generic import utils

class Chirp(experiment.Stimulus):
    def configuration(self):
        self.FREQ_START=0.1#Hz
        self.FREQ_END=2#Hz
        self.TIME_INTERVAL=60*5#sec
        self.COLOR=1.0
        self.WAIT=0.5#wait time in seconds at beginning and end of stimulus
        self.REPEATS=1
        self.BACKGROUND=0.0
        self.FS=60#frequency which makes the time vector. should be > self.FREQ_END * 2 and < frame rate of the projector
        self.CONTINIOUS_BOOL=True#colors will be between 0 and 1 unless it is False and colors will be 0 or 1
#Do not edit below this!

    def make_chirp(self):
        T = self.TIME_INTERVAL
        f0 = self.FREQ_START
        f1 = self.FREQ_END
        fs = self.FS
        t = numpy.arange(0, T+1.0/fs, 1.0/fs)
        chirp_signal = numpy.sin(2*numpy.pi*(f0*t + (f1-f0)/(2*T)*t**2))
        if self.CONTINIOUS_BOOL:
            chirp_signal = (chirp_signal+1)/2
        else:
            chirp_signal[chirp_signal<0]=0
            chirp_signal[chirp_signal>0]=1
        self.CHIRP_SIGNAL=chirp_signal
        self.CHIRP_SIGNAL_TIME=t

    def prepare(self):
        self.make_chirp()
        print(self.screen.frame_rate)
 
    def run(self):
        self.show_fullscreen(color=self.BACKGROUND, duration=self.WAIT)
        for r in range(self.REPEATS):
            for frame_no in range(len(self.CHIRP_SIGNAL_TIME)):
                color = self.CHIRP_SIGNAL[frame_no]
                self.block_start(('on',))
                self.show_fullscreen(color=color, duration=1/self.FS)
                self.block_end()
                if self.abort:
                    break
        self.show_fullscreen(color=self.BACKGROUND, duration=self.WAIT)
