from crewai import Agent, Crew, Process, Task, LLM
from crewai_tools import FileReadTool
import logging
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from ai_coding_team.tools.custom_tool import (
    PlanProjectTool, ArchitectureWriterTool, GenerateCodeTool,
    BuildProjectTool, GenerateTestsTool, RunTestsTool,
    WriteDocsTool, ReviewRepoTool
)
import yaml  
import os     

logger = logging.getLogger(__name__)

# ---------- Load Team Rules ----------  # 
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")
team_rules_path = os.path.join(CONFIG_DIR, "team_rules.yaml")

try:
    with open(team_rules_path, "r", encoding="utf-8") as f:
        TEAM_RULES = yaml.safe_load(f).get("rules", "")
except Exception as e:
    TEAM_RULES = ""
    logger.warning(f"team_rules.yaml not loaded: {e}")


@CrewBase
class AiCodingTeam():
    """AiCodingTeam crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    llm = LLM(model="ollama/qwen2.5-coder:7b", base_url="http://localhost:11434")

    # ---------- helper to apply rules ----------  # 
    def _inject_rules(self, config: dict) -> dict:
        """Append team rules to system prompt without modifying agents.yaml."""
        new_config = config.copy()

        # Ensure system prompt exists
        system_prompt = new_config.get("system", "")
        if system_prompt and TEAM_RULES:
            new_config["system"] = f"{system_prompt}\n\n--- TEAM RULES ---\n{TEAM_RULES}"
        elif TEAM_RULES:
            new_config["system"] = f"--- TEAM RULES ---\n{TEAM_RULES}"

        return new_config

    @agent
    def product_manager(self) -> Agent:
        logger.info("product_manager initialized")
        return Agent(
            config=self._inject_rules(self.agents_config['product_manager']), 
            llm=self.llm,
            tools=[FileReadTool(), PlanProjectTool()],
            verbose=True,
            force_tool_execution=True
        )

    @agent
    def architect(self) -> Agent:
        logger.info("architect initialized")
        return Agent(
            config=self._inject_rules(self.agents_config['architect']),         
            llm=self.llm,
            tools=[ArchitectureWriterTool()],
            verbose=True,
            force_tool_execution=True
        )

    @agent
    def backend_engineer(self) -> Agent:
        logger.info("backend_engineer initialized")
        return Agent(
            config=self._inject_rules(self.agents_config['backend_engineer']), 
            llm=self.llm,
            tools=[BuildProjectTool(), GenerateCodeTool()],
            verbose=True,
            allow_code_execution=True,
            code_execution_mode="safe",
        )

    @agent
    def frontend_engineer(self) -> Agent:
        logger.info("frontend_engineer initialized")
        return Agent(
            config=self._inject_rules(self.agents_config['frontend_engineer']), 
            llm=self.llm,
            tools=[BuildProjectTool(), GenerateCodeTool()],
            verbose=True,
            allow_code_execution=True,
            code_execution_mode="safe",
        )

    @agent
    def qa_engineer(self) -> Agent:
        logger.info("qa_engineer initialized")
        return Agent(
            config=self._inject_rules(self.agents_config['qa_engineer']), 
            llm=self.llm,
            tools=[GenerateTestsTool(), RunTestsTool()],
            verbose=True,
            allow_code_execution=True,
            code_execution_mode="safe",
        )

    @agent
    def tech_writer(self) -> Agent:
        logger.info("tech_writer initialized")
        return Agent(
            config=self._inject_rules(self.agents_config['tech_writer']), 
            tools=[WriteDocsTool()],
            llm=self.llm,
            verbose=True,
        )

    @agent
    def reviewer(self) -> Agent:
        logger.info("reviewer initialized")
        return Agent(
            config=self._inject_rules(self.agents_config['reviewer']), 
            tools=[ReviewRepoTool()],
            llm=self.llm,
            verbose=True,
        )

    @agent
    def devops_engineer(self) -> Agent:
        logger.info("devops_engineer initialized")
        return Agent(
            config=self._inject_rules(self.agents_config['devops_engineer']),
            tools=[BuildProjectTool(), GenerateCodeTool(), ReviewRepoTool(), WriteDocsTool()],
            llm=self.llm,
            verbose=True,
        )

    @task
    def collect_requirements(self) -> Task:
        return Task(config=self.tasks_config['collect_requirements'])

    @task
    def architect_project(self) -> Task:
        return Task(config=self.tasks_config['architect_project'])

    @task
    def build_backend(self) -> Task:
        return Task(config=self.tasks_config['build_backend'])

    @task
    def build_frontend(self) -> Task:
        return Task(config=self.tasks_config['build_frontend'])

    @task
    def quality_assurance(self) -> Task:
        return Task(config=self.tasks_config['quality_assurance'])

    @task
    def build_infrastructure_and_ci_cd(self) -> Task:
        return Task(config=self.tasks_config['build_infrastructure_and_ci_cd'])

    @task
    def write_docs(self) -> Task:
        return Task(config=self.tasks_config['write_docs'])

    @task
    def review_repo(self) -> Task:
        return Task(config=self.tasks_config['review_repo'])
    
    @crew
    def crew(self) -> Crew:
        logger.info("crew created")
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            tracing=True
        )
