import sys

def get_error_details(error, sys_module:sys):
    _,_,exc_tb=sys_module.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_msg=f"Error occured in the script: {file_name}\nLine no.: {exc_tb.tb_lineno}\nError message: {str(error)}"

    return error_msg

class CustomException(Exception):
    def __init__(self, error, sys_module:sys):
        super().__init__(error)
        self.error_msg=get_error_details(error, sys_module)

    def __str__(self):
        return self.error_msg
