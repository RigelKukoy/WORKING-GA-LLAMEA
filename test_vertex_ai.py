import os
from dotenv import load_dotenv
from iohblade.llm import Gemini_LLM

def test_vertex():
    load_dotenv()
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "asia-southeast1")
    
    print(f"Testing Vertex AI with:")
    print(f"  Project: {project}")
    print(f"  Location: {location}")
    
    if not project:
        print("ERROR: GOOGLE_CLOUD_PROJECT not found in .env")
        return

    try:
        # Using the default model in the constructor if none provided
        llm = Gemini_LLM(project=project, location=location, model="gemini-2.0-flash")
        
        prompt = "Say 'Vertex AI is working!' if you can hear me."
        print(f"\nSending prompt: {prompt}")
        
        response = llm.query([{"role": "user", "content": prompt}])
        
        print(f"\nResponse from LLM:\n{response}")
        
    except Exception as e:
        print(f"\nAn error occurred during testing:\n{e}")

if __name__ == "__main__":
    test_vertex()
