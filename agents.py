from crewai import Agent
from crewai_tools import WebsiteSearchTool, SerperDevTool
from tools.search_tools import SearchTools
from tools.search_tools import search_tool
from langchain.llms import OpenAI, Ollama
from langchain_community.llms import OpenAI, Ollama
from langchain_openai import ChatOpenAI

web_search_tool = WebsiteSearchTool()
serper_dev_tool = SerperDevTool()


class AINewsLetterAgents():
    
    def __init__(self):
        self.OpenAIGPT35 = ChatOpenAI(model="gpt-3.5", temperature=0.7)
        self.OpenAIGPT4 = ChatOpenAI(model="gpt-4", temperature=0.7)
        self.Ollama = Ollama(model="crewai-llama2",base_url="http://localhost:11434/v1",api_key="NA")
    

    def editor_agent(self):
        return Agent(
            role='Editor',
            goal='Oversee the creation of the AI Newsletter',
            backstory="""With a keen eye for detail and a passion for storytelling, you ensure that the newsletter
            not only informs but also engages and inspires the readers. """,
            allow_delegation=True,
            verbose=True,
            llm=self.Ollama,
            max_iter=15
        )

    def news_fetcher_agent(self):
        return Agent(
            role='NewsFetcher',
            goal='Fetch the top AI news stories for the day',
            backstory="""As a digital sleuth, you scour the internet for the latest and most impactful developments
            in the world of AI, ensuring that our readers are always in the know.""",
            tools=[web_search_tool,serper_dev_tool],
            verbose=True,
            llm=self.Ollama, 
            allow_delegation=True,
        )

    def news_analyzer_agent(self):
        return Agent(
            role='NewsAnalyzer',
            goal='Analyze each news story and generate a detailed markdown summary',
            backstory="""With a critical eye and a knack for distilling complex information, you provide insightful
            analyses of AI news stories, making them accessible and engaging for our audience.""",
            tools=[web_search_tool,serper_dev_tool],
            verbose=True,
            llm=self.Ollama,
            allow_delegation=True,
        )

    def newsletter_compiler_agent(self):
        return Agent(
            role='NewsletterCompiler',
            goal='Compile the analyzed news stories into a final newsletter format',
            backstory="""As the final architect of the newsletter, you meticulously arrange and format the content,
            ensuring a coherent and visually appealing presentation that captivates our readers. Make sure to follow
            newsletter format guidelines and maintain consistency throughout.""",
            verbose=True,
            llm=self.Ollama,
        )