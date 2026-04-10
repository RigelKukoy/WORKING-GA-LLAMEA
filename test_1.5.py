import os
from dotenv import load_dotenv
from iohblade.llm import Gemini_LLM

def test():
    load_dotenv()
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "asia-southeast1")
    
    print(f"Testing Vertex AI with:")
    print(f"  Project: {project}")
    print(f"  Location: {location}")
    print(f"  Model: gemini-1.5-flash")
    
    try:
        llm = Gemini_LLM(project=project, location=location, model="gemini-1.5-flash")
        prompt = "Say 'Gemini 1.5 Flash is working!' if you can hear me."
        print(f"\nSending prompt: {prompt}")
        response = llm.query([{"role": "user", "content": prompt}])
        print(f"\nResponse from LLM:\n{response}")
    except Exception as e:
        print(f"\nAn error occurred:\n{e}")

if __name__ == "__main__":
    test()
