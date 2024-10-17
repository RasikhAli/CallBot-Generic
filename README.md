# CallBot-Generic

### Define the model name
model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"

### Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

### Save the model and tokenizer locally
tokenizer.save_pretrained("LLM/paraphrase-MiniLM-L6-v2")
model.save_pretrained("LLM/paraphrase-MiniLM-L6-v2")
