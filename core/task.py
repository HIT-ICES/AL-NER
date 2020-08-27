import re


class Task:
    def __init__(self, task_def, logger):
        task_name = task_def.__name__
        if re.search("^[a-zA-Z]\w+$", task_name):
            self._name = task_name
        else:
            error_tips = f"[task.py]--[ERROR]: task name \" {task_name} \" is invalid (should match [a-zA-Z][a-zA-Z0-9_]+)!"
            logger.error(error_tips)
            raise Exception(error_tips)

        self._name = task_name
        self._task_def = task_def

    @property
    def name(self):
        return self._name
    
    def submit_task(self):
        self._task_def()
        return
