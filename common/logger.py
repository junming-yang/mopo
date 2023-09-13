import os
import atexit


class Logger:
    def __init__(self, writer, output_fname="progress.txt", log_path="log"):
        self.writer = writer
        self.log_path = self.writer.get_logdir()
        self.output_file = open(os.path.join(self.writer.get_logdir(), output_fname), 'w')
        self.log_path = log_path
        atexit.register(self.output_file.close)

    def record(self, tag, scalar_value, global_step, printed=True):
        self.writer.add_scalar(tag, scalar_value, global_step)
        if printed:
            info = f"{tag}: {scalar_value:.3f}"
            print("\033[1;32m [info]\033[0m: " + info)
            self.output_file.write(info + '\n')

    def print(self, info):
        print("\033[1;32m [info]\033[0m: " + info)
        self.output_file.write(info + '\n')
