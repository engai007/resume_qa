from app.agents.resume_parser import ResumeParserAgent

# Every agent in the system. Add new agents here.
AGENTS = {agent.name: agent for agent in [ResumeParserAgent]}
