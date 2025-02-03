import sys

def error_msg_detail(error, error_detail:sys):
    _, _, excp_detail = error_detail.exc_info()

    file_name = excp_detail.tb_frame.f_code.co_filename
    line_number = excp_detail.tb_lineno

    error_msg = "Error occured in python script name [{0}] line number [{1}] and error message [{2}]".format(file_name, line_number, str(error))

    return error_msg

class CustomException(Exception):
    def __init__(self, error_msg, error_detail:sys):
        super().__init__(error_msg)
        self.error_msg = error_msg_detail(error_msg, error_detail=error_detail)

    def __str__(self):
        return self.error_msg
    