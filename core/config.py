import os
import configparser


class Config(configparser.SafeConfigParser):
    def __init__(self):
        configparser.SafeConfigParser.__init__(self)
        
    def parse_files(self, config_files):
        """
        Parse configuration file
        :params config_files:
        """
        self.optionxform = str
        for config_file in config_files:
            self.read_file(config_file)
            
        return

    def param(self, section, option, required=True, type='string'):
        """
        Parse the value of the option under the given section
        :param section
        :param option
        :param required:default True
        :param type:default string (It may need to be improved)
        """
        section_choose = section

        if not self.has_section(section):
            section = 'DEFAULT'

        if self.has_option(section, option):
            try:
                if type == 'int':
                    return self.getint(section, option)
                elif type == 'float':
                    return self.getfloat(section, option)
                elif type == 'boolean':
                    return self.getboolean(section, option)
                elif type == 'string':
                    return self.get(section, option)
                elif type == 'filepath':
                    value = os.path.expandvars(self.get(section, option))
                    if os.path.isfile(value):
                        return value
                    else:
                        raise Exception(f'[config.py]--[ERROR]: File path: \"{value}\" does not exist or is not a valid regular file!')
                elif type == 'dirpath':
                    value = os.path.expandvars(self.get(section, option))
                    if os.path.isdir(value):
                        return value
                    else:
                        raise Exception(f'[config.py]--[ERROR]: Directory path:\"{value}\" does not exist or is not a valid directory!')
                else:
                    raise Exception("Unknown parameter type '" + type + "'")
            except Exception as e:
                raise Exception(f'parameter \"{section}.{option}\" is invalid !\n')
        elif required:
            raise Exception(f'parameter \"{section_choose}.{option}\" is not defined in config file(s)!')
        else:
            return ""
