import os
import atexit


class Logger:
    def __init__(self, writer, output_fname="progress.txt"):
        self.writer = writer
        self.output_file = open(os.path.join(self.writer.get_logdir(), output_fname), 'w')
        atexit.register(self.output_file.close)

    def record(self, tag, scalar_value, global_step, printed=True):
        self.writer.add_scalar(tag, scalar_value, global_step)
        if printed:
            info = f"{tag}: {scalar_value:.3f}"
            print(info)
            self.output_file.write(info + '\n')
    
    def print(self, info):
        print(info)
        self.output_file.write(info + '\n')
