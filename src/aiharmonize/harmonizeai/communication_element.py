from pydantic import BaseModel, Field, validator


class FunctionPoint(BaseModel):
    """Function Point Model"""
    description: str = Field(description="Briefly describes the functionality of this function.")
    param: list[str] = Field(description="Describes the definition of each parameter.")
    return_value: str = Field(description="Describes the return value of the function.")
    function_name: str = Field(description="Displays the name of the function.")
    function_type: str = Field(description="Displays the type of the function.")
    function_file: str = Field(description="Displays the file of the function.")
    function_module: str = Field(description="Displays the module of the function.")
    function_class: str = Field(description="Displays the class of the function.")


class FunctionPoints(BaseModel):
    """Function Points Model"""
    function_points: list[FunctionPoint] = Field(description="List of function points.")
