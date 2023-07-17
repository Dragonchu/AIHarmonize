from pydantic import BaseModel, Field, validator


class FunctionPoint(BaseModel):
    """Function Point Model"""
    description: str = Field(description="Briefly describes the functionality of this function.")
    param: list[str] = Field(description="Describes the definition of each parameter.")
    return_value: str = Field(description="Describes the return value of the function.")
    function_name: str = Field(description="Displays the name of the function.")
    function_class: str = Field(description="Displays the class of the function.")
    retain: bool = Field(description="Whether to retain the function.",default=True)


class FunctionPoints(BaseModel):
    """Function Points Model"""
    function_points: list[FunctionPoint] = Field(description="List of function points.")
    
class MergePlan(BaseModel):
    """Merge Plan Model"""
    merge_plan: list[str] = Field(description="The steps of merge two class into one class.")
