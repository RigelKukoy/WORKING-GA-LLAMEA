import os
from dotenv import load_dotenv
from google import genai

def main():
    load_dotenv()
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    quota_project = os.getenv("GOOGLE_CLOUD_QUOTA_PROJECT") or project
    os.environ["GOOGLE_CLOUD_QUOTA_PROJECT"] = quota_project
    
    # Check Taiwan (Taiwan) and Tokyo (Tokyo) as alternatives
    for loc in ["asia-east1", "asia-northeast1", "asia-southeast1"]:
        print(f"\nChecking region: {loc}")
        client = genai.Client(vertexai=True, project=project, location=loc)
        try:
            found = []
            for model in client.models.list():
                if 'gemini' in model.name:
                    found.append(model.name)
            if found:
                print(f"Found {len(found)} Gemini models. Top 3: {found[:3]}")
            else:
                print("No Gemini models found.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
