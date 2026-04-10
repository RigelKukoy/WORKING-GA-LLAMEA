import os
from dotenv import load_dotenv
from google import genai

def main():
    load_dotenv()
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = "asia-southeast1" # Singapore
    quota_project = os.getenv("GOOGLE_CLOUD_QUOTA_PROJECT") or project
    
    os.environ["GOOGLE_CLOUD_QUOTA_PROJECT"] = quota_project
    
    print(f"Project: {project}, Location: {location}, Quota: {quota_project}")
    
    client = genai.Client(vertexai=True, project=project, location=location)
    
    print("Listing ALL models in asia-southeast1...")
    try:
        found = False
        for model in client.models.list():
            if 'gemini' in model.name:
                print(f"Model: {model.name}")
                found = True
        if not found:
            print("No Gemini models found in this region.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
