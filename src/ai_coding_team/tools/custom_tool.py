from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from ai_coding_team.tools.generation import GenerationOps, read_file
import logging
from typing import Type

logger = logging.getLogger(__name__)

# Model for PlanProjectTool
class PlanProjectInput(BaseModel):
    requirements_path: str = Field(..., description="Path to the file containing project requirements.")
    project_name: str = Field(..., description="Name of the project.")
    output_dir: str = Field(..., description="Directory where the project plan should be saved.")
    file_name: str = Field(..., description="Name of the file to save the project plan.")
    content: str = Field(..., description="Content of the project plan to be written.")

class PlanProjectTool(BaseTool):
    name: str = "PlanProjectTool"
    description: str = (
        "Useful when you want to plan a project. The input to this tool should be a detailed description of the project requirements."
    )
    args_schema: Type[BaseModel] = PlanProjectInput

    def _run(self, requirements_path: str, project_name: str, output_dir: str, file_name: str, content: str) -> str:        
        logger.info("PlanProjectTool _run started")
        text = read_file(requirements_path)
        GenerationOps.write_plan(text, project_name, output_dir, file_name, content)
        logger.info("PlanProjectTool _run finished")
        return text

# Model for ArchitectureWriterTool
class ArchitectureWriterInput(BaseModel):
    project_name: str = Field(..., description="Name of the project.")
    output_dir: str = Field(..., description="Directory where the architecture plan should be saved.")
    file_name: str = Field(..., description="Name of the file to save the architecture plan.")
    content: str = Field(..., description="Content of the architecture plan to be written.")

class ArchitectureWriterTool(BaseTool):
    name: str = "ArchitectureWriterTool"
    description: str = "Writes the architecture output provided by the Architect agent into a file."
    args_schema: Type[BaseModel] = ArchitectureWriterInput

    def _run(self, project_name: str, output_dir: str, file_name: str, content: str):
        logger.info("ArchitectureWriterTool _run started")
        result = GenerationOps.write_architecture(project_name, output_dir, file_name, content)
        logger.info("ArchitectureWriterTool _run finished")
        return result

# Model for GenerateCodeTool
class GenerateCodeInput(BaseModel):
    module_name: str = Field(..., description="Name of the module to be created.")
    project_name: str = Field(..., description="Name of the project.")
    output_dir: str = Field(..., description="Directory where the code file should be saved.")
    code_stub: str = Field(..., description="Stub code to be written into the file.")

class GenerateCodeTool(BaseTool):
    name: str = "GenerateCodeTool"
    description: str = "Create a stub code file that the LLM-based agent will fill."
    args_schema: Type[BaseModel] = GenerateCodeInput

    def _run(self, module_name: str, project_name: str, output_dir: str, code_stub: str):
        logger.info("GenerateCodeTool _run started")
        result = GenerationOps.write_code(module_name, project_name, output_dir, code_stub)
        logger.info("GenerateCodeTool _run finished")
        return result

# Model for BuildProjectTool
class BuildProjectInput(BaseModel):
    output_dir: str = Field(..., description="Directory where the project should be built.")

class BuildProjectTool(BaseTool):
    name: str = "BuildProjectTool"
    description: str = "Ensure project output directory exists."
    args_schema: Type[BaseModel] = BuildProjectInput

    def _run(self, output_dir: str):
        logger.info("BuildProjectTool _run started")
        result = GenerationOps.create_project_folder(output_dir)
        logger.info("BuildProjectTool _run finished")
        return result

# Model for GenerateTestsTool
class GenerateTestsInput(BaseModel):
    project_name: str = Field(..., description="Name of the project.")
    module_name: str = Field(..., description="Name of the module to generate tests for.")
    output_dir: str = Field(..., description="Directory where the test file should be saved.")
    test_stub: str = Field("", description="Stub code to be written into the test file.")

class GenerateTestsTool(BaseTool):
    name: str = "GenerateTestsTool"
    description: str = "Creates placeholder test files for modules so the QA agent can expand them."
    args_schema: Type[BaseModel] = GenerateTestsInput

    def _run(self, project_name: str, module_name: str, output_dir: str, test_stub: str = ""):
        logger.info("GenerateTestsTool _run started")
        result = GenerationOps.generate_tests(module_name, project_name, output_dir, test_stub)
        logger.info("GenerateTestsTool _run finished")
        return result

# Model for RunTestsTool
class RunTestsInput(BaseModel):
    repo_path: str = Field(..., description="Path to the repository where tests should be run.")

class RunTestsTool(BaseTool):
    name: str = "RunTestsTool"
    description: str = "Runs tests (pytest) and returns results for QA evaluation."
    args_schema: Type[BaseModel] = RunTestsInput

    def _run(self, repo_path: str):
        logger.info("RunTestsTool _run started")
        result = GenerationOps.run_tests(repo_path)
        logger.info("RunTestsTool _run finished")
        return result


# Model for WriteDocsTool
class WriteDocsInput(BaseModel):
    project_name: str = Field(..., description="Name of the project.")
    output_dir: str = Field(..., description="Directory where the documentation should be saved.")

class WriteDocsTool(BaseTool):
    name: str = "WriteDocsTool"
    description: str = "Create documentation placeholders."
    args_schema: Type[BaseModel] = WriteDocsInput

    def _run(self, project_name: str, output_dir: str, content: str = ""):
        logger.info("WriteDocsTool _run started")
        result = GenerationOps.write_docs(project_name, output_dir, content)
        return result

# Model for ReviewRepoTool
class ReviewRepoInput(BaseModel):
    repo_path: str = Field(..., description="Path to the repository to be reviewed.")

class ReviewRepoTool(BaseTool):
    name: str = "ReviewRepoTool"
    description: str = "Placeholder tool for reviewing a repository. The agent will reason using LLM."
    args_schema: Type[BaseModel] = ReviewRepoInput  

    def _run(self, repo_path: str):
        logger.info("ReviewRepoTool _run started")
        result = f"Review process initialized for: {repo_path}"
        logger.info("ReviewRepoTool _run finished")
        return result
