import os
from huggingface_hub import InferenceClient

#os.environ["HF_TOKEN"] = "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"

client = InferenceClient(
    api_key=os.environ["HF_TOKEN"],
)

sentences = "add like flavor dishes fresh herbs depth dish bring thyme. team championship fans goal match soccer win stadium crucial winning. natural vegetables brings flavor flavors roasting pinch salt enhances ingredients. team's success opposing championship morale player win performance soccer tough. players need soccer must training stay adaptable succeed work achieve. goal scored scoring match player team's single score fans thrill. dough every bakery time pastry goalkeeper work saves goals hard. kitchen cook like home serious memories made baking well stocked. bread perfect freshly baking baked oven football sourdough aroma day. match difference well make coach's teams tactics winning electric intense. dish flavorful meal elevate balance flavors great ordinary extraordinary well. meat pan searing hot crust flavor adds creates that's full. yeast flour dough bakers crispy kneading rising creates interior chicken. cooking requires art patience practice baking love skill scratch labor. soccer game fans sport strategy requires match beautiful teams skill. sourdough starter league needs regular champions process creating time consuming. new decision heights referee's fans game harsh us meal pinch. spices herbs flavor right make break recipe de provence french. midfielder's skills vision footballers passing ability control game spend field. championship recipe always rain face showers tonight matches winner cayenne"


instruction = """You find words that construct sentences that might be syntactically incorrect.
Cluster these sentences by their semantic meaning.
Return a JSON object with cluster labels as keys and lists of sentences as values. Use meaningful cluster labels based on the common themes of the sentences in each cluster. 
Be concise but descriptive in your cluster labels.
Some sentenced may be impure and contain multiple themes, so assign them to a cluster named 'Mixed Themes'. 
Return only the JSON object, without any additional text or explanation.
"""

completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-32B-Instruct:featherless-ai",
    messages=[
        {
            "role": "user",
            "content": f"{instruction}\n\nSentences:\n{sentences}"
        }
    ],
)

print(completion.choices[0].message.content)