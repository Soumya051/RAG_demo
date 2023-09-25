from sentence_transformers import SentenceTransformer
import regex as re

class DocumentParser:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.data = ''
        # self.url = 'https://api.openai.com/v1/embeddings'
        # self.auth = HTTPBasicAuth('apikey', os.environ.get('openai_api_key'))

        # self.model_name = 'roberta-base'
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # self.model = AutoModelForTextEncoding.from_pretrained(self.model_name)
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # self.classification_model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)

    def read_file(self, filepath):
        with open(filepath, 'r') as f:
            self.data = f.read()
        data = {'text': self.data}
        self.clean_data()
        data['embedding'] = self.get_embedding()

        return data

    def clean_data(self):
        self.data = re.sub(r'[^a-zA-Z .]', '', self.data)
        self.data = re.sub(r' +', ' ', self.data)

    def get_embedding(self, data = ''):
        if data:
          self.data = data
          self.clean_data()
        # output = requests.post(url = self.url, json = {'model':"text-embedding-ada-002", 'input':self.data}, auth=self.auth).json()
        # embedding = output['data'][0]['embedding']

        # tokenized_text = self.tokenizer(self.data, return_tensors= 'pt')
        # embedding = self.model(**tokenized_text).last_hidden_state
        # return embedding[:, 0, :].detach().numpy()[0]

        embedding = self.model.encode(self.data)
        return embedding