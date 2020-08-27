# Python Standard Modules
import argparse
import os


def get_pipeline_arg_parser(epilog):
    argparser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, epilog=epilog, conflict_handler='resolve')

    # Common options for all pipelines
    argparser.add_argument("--help", help="show detailed description of pipeline and steps", action="store_true")
    argparser.add_argument("-c", "--config", help="config INI-style list of files; config parameters are overwritten based on files order", nargs="+", type=argparse.FileType(mode="r"))
    argparser.add_argument("-t", "--tasks", help="task range e.g. '1-5', '3,6,7', '2,4-8'")
    argparser.add_argument("-l", "--log", help="log level (default: info)", choices=["debug", "info", "warning", "error", "critical"], default="info")
    argparser.add_argument("--project", help="project name of current run")
    return argparser