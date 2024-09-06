from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from openai import AzureOpenAI

load_dotenv()  # take environment variables from .env.

## Function to load OpenAI model and get responses
def get_openai_response(question):
    llm = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )
    response = llm.c(question)
    return response


training_text = """
Hon Luu, a highly skilled Data Scientist, has built a successful career over the years, demonstrating expertise in programming languages 
and frameworks such as Python, R, SAS, SQL, VBA, DAX, Git, scikit-learn, pandas, and TensorFlow. With a strong background in data, ETL, 
and visualization tools, including Advanced Excel, VBA, PowerBI, Quicksight, Plotly, Airflow, Spark, and various AWS services, 
he has consistently delivered valuable insights through his proficiency in machine learning and modeling techniques, 
as well as statistics and probability. Starting his journey as a Senior Analyst at Kemper and Hallmark Insurance from 2011 to 2013, Hon played a crucial role in supporting product managers, executing pricing decisions, and analyzing performance data. During this time, he evaluated premium and loss performance, conducted mix and competitor analysis, and proposed findings to senior management. Leveraging his skills in SQL, Excel, and VBA, he developed performance-tracking reports that provided valuable insights to stakeholders.
Continuing his career, Hon joined Liberty Mutual Insurance as a Data Scientist from 2013 to 2015. Here, he showcased his ability to develop, validate, and iterate statistical models within an ambiguous environment. He successfully narrowed down 20+ independent variables to 5, resulting in improved model accuracy. Leveraging machine learning and predictive analytics, he delivered measurable insights such as identifying top purchase combinations using association data mining metrics and surfacing new insights on price change sensitivity using propensity score matching. Hon spearheaded deep-dive customer retention analysis through survival analysis techniques, which resulted in additional premium revenue and positively impacted a significant number of customers. He also led multiple multi-month-long projects, collaborating closely with cross-functional teams to ensure successful pricing implementation.
In 2015, Hon transitioned to the role of Program Manager II at Microsoft Corporation, where he managed growth and monetization for the Bing team. During his tenure until 2017, he excelled in automating query and dashboard development, leading to a remarkable 50% improvement in run and load time. His work provided insightful takeaways regarding the programmatic funnel, contributing to the identification of new opportunities within emerging markets.
From 2017 to 2019, Hon served as a Consulting Manager at PricewaterhouseCoopers (PwC) in the United States. Here, he provided valuable support to actuarial teams and developed PowerBI dashboards for automation and analysis. His successes included developing a PowerBI dashboard to assess mortality rates for a life insurance client and automating P&L analysis, resulting in an impressive 80% increase in efficiency. Hon's ability to pitch modeling proposals to Fortune 50 companies showcased his talent for driving impactful insights.
Currently, Hon holds the position of Product Lead at AlikeAudience, a role he has held since 2019. In this role, he has successfully developed a comprehensive multi-year product roadmap, enhanced product offerings in the APAC market, and effectively managed the entire product life cycle. His contributions extend to gaining alignment with cross-functional teams on priorities, suggesting three new verticals to enter the market, and coaching and managing an analyst to a leadership role. Hon's expertise and leadership have been instrumental in the day-to-day operations of the company, where he is acting as the head of product.
Hon Luu holds a Master of Science in Data Science from Northwestern University (2016 - 2018) and a Bachelor of Arts in Mathematics from the University of Oklahoma (2007 - 2011).
"""


input_text = "Describe hon's experience"

input_text += "\n" + training_text
response = get_openai_response(input_text)
print(response)
