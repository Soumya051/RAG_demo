from transformers import T5Tokenizer, T5ForConditionalGeneration

class Generator:
  def __init__(self, model_name: str):
    self.model_name = model_name
    self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
    self.generator = T5ForConditionalGeneration.from_pretrained(self.model_name)
  
  def run_query(self, query: str, augment_data = ''):
    if augment_data:
      augmented_query = augment_data + ' . ' + query
    else:
      augmented_query = query
    input_ids = self.tokenizer(augmented_query, return_tensors = 'pt').input_ids
    output = self.generator.generate(input_ids)
    
    return self.tokenizer.decode(output[0], skip_special_tokens = True)