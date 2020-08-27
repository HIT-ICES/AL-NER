import os
import re
import sys
import textwrap
import datetime
sys.path.append("..")

# AL-NER-DEMO Modules
from core.config import Config
from core.task import Task
from utils.logger import Logger
from utils.arg_parser import get_pipeline_arg_parser


class Pipeline(object):
    """
    Pipeline
    ===============================
    """
    def __init__(self):
        self._timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self._args = self.argparser.parse_args()
        self._config = Config()
        if self.args.help:
            print("""\
[TOC]

{pipeline_doc}

Usage
-----
```
#!text

{help}
```
{task_doc}
""".format(
            pipeline_doc=textwrap.dedent(self.__doc__) if self.__doc__ else "",
            help=self._argparser.format_help(),
            task_doc="\n".join([str(idx + 1) + "- " + task.__name__ + "\n" + "-" * len(str(idx + 1) + "- " + task.__name__) + (textwrap.dedent(task.__doc__) if task.__doc__ else "") for idx, task in enumerate(self.tasks)])
        ))
            self._argparser.exit()
        else:
            if self.args.config:
                self.config.parse_files(self.args.config)
                self.logger = Logger(__name__, config=self.config)
            else:
                self.argparser.error("argument -c/--config is required!")
            
            self._task_list = [Task(task, self.logger) for task in self.tasks]
            
            if self.args.tasks:
                if re.search("^\d+([,-]\d+)*$", self.args.tasks):
                    self._task_range = [self.task_list[i - 1] for i in parse_task_range(self.args.tasks)]
                else:
                    error_tips = f"[pipeline.py]--[ERROR]: task range \" {self.args.tasks} \" is invalid (should match \d+([,-]\d+)*)!"
                    self.logger.error(error_tips)
                    raise Exception(error_tips)
            else:
                error_tips = "[pipeline.py]--[ERROR]: argument -t/--tasks is required!"
                self.logger.error(error_tips)
                raise Exception(error_tips)

            if self.args.project:
                if not self.args.project.isdigit(): 
                    error_tips = f"[pipeline.py]--[ERROR]: project id \" {self.args.project} \" is invalid (should be 5 bit int)!"
                    self.logger.error(error_tips)
                    raise Exception(error_tips)
                self.project = self.args.project
            else:
                self.logger.warning("[pipeline.py]--[WARNING]: argument --project is not given!")

            for task_choose in self.task_range:
                task_choose.submit_task()

    @property
    def argparser(self):
        if not hasattr(self, "_argparser"):
            epilog = """\
Tasks:
------
{tasks}
""".format(tasks="\n".join([str(idx + 1) + "- " + task.__name__  for idx, task in enumerate(self.tasks)]))        
            
            self._argparser = get_pipeline_arg_parser(epilog)
            return self._argparser
       
    @property
    def config(self):
        self._config = Config()
        return self._config

    @property
    def args(self):
        return self._args

    @property
    def config(self):
        return self._config

    @property
    def timestamp(self):
        return self._timestamp


    @property
    def tasks(self):
        # Needs to be defined in pipeline child class
        raise NotImplementedError

    @property
    def task_list(self):
        return self._task_list

    @property
    def task_range(self):
        return self._task_range


def parse_task_range(task_range):
    '''
    Return a range list given a string.
    :param astr:The value of the parameter: -s
    e.g. parse_task_range('1,3,5-7') returns [1, 3, 5, 6, 7]
    '''
    result = set()
    for part in task_range.split(','):
        x = part.split('-')
        result.update(range(int(x[0]), int(x[-1]) + 1))
    return sorted(result)