import os
import shutil
import numpy as np
import torch
from transformers import RobertaTokenizer
import sys
import importlib.util

# Import the model class from try.py
try:
    # Use importlib.util to load the module without using the 'try' keyword
    spec = importlib.util.spec_from_file_location("travel_model", "try.py")
    travel_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(travel_model)
    
    # Get the DestinationRecommender class
    DestinationRecommender = travel_model.DestinationRecommender
except Exception as e:
    print(f"Error: Could not import DestinationRecommender from try.py: {e}")
    print("Make sure try.py is in the current directory and contains the DestinationRecommender class.")
    sys.exit(1)

def update_model():
    """
    Updates the model with new destination embeddings
    """
    print("Starting model update...")
    
    # Check if the new embeddings file exists
    if not os.path.exists('destination_embeddings.npy'):
        print("Error: destination_embeddings.npy not found.")
        print("Please make sure the file is in the current directory.")
        return False
    
    # Load the new embeddings
    try:
        print("Loading new destination embeddings...")
        new_embeddings = np.load('destination_embeddings.npy')
        print(f"Loaded embeddings with shape: {new_embeddings.shape}")
    except Exception as e:
        print(f"Error loading destination_embeddings.npy: {e}")
        return False
    
    # Output directory
    model_dir = './model_output/'
    
    # Create the directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory: {model_dir}")
    
    # Check if the model exists
    model_path = os.path.join(model_dir, 'roberta_destination_model.pt')
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("The model needs to exist before updating embeddings.")
        return False
    
    # Back up the old embeddings if they exist
    embeddings_path = os.path.join(model_dir, 'destination_embeddings.npy')
    if os.path.exists(embeddings_path):
        backup_path = os.path.join(model_dir, 'destination_embeddings.backup.npy')
        try:
            shutil.copy2(embeddings_path, backup_path)
            print(f"Backed up old embeddings to {backup_path}")
        except Exception as e:
            print(f"Warning: Could not back up old embeddings: {e}")
    
    # Copy the new embeddings to the model directory
    try:
        shutil.copy2('destination_embeddings.npy', embeddings_path)
        print(f"Updated embeddings in {embeddings_path}")
    except Exception as e:
        print(f"Error copying new embeddings: {e}")
        return False
    
    # At this point, we've successfully updated the embeddings
    
    # Now check if the tokenizer exists
    tokenizer_dir = os.path.join(model_dir, 'tokenizer')
    if not os.path.exists(tokenizer_dir):
        print("Warning: Tokenizer directory not found. Creating it...")
        
        try:
            os.makedirs(tokenizer_dir, exist_ok=True)
            
            # If we have tokenizer files in the current directory, copy them
            for file_name in ['special_tokens_map.json', 'tokenizer_config.json', 'vocab.json']:
                if os.path.exists(os.path.join('tokenizer', file_name)):
                    shutil.copy2(
                        os.path.join('tokenizer', file_name),
                        os.path.join(tokenizer_dir, file_name)
                    )
                    print(f"Copied {file_name} to tokenizer directory")
            
            if not os.path.exists(os.path.join(tokenizer_dir, 'vocab.json')):
                print("Warning: vocab.json not found. Using RobertaTokenizer default.")
                tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
                tokenizer.save_pretrained(tokenizer_dir)
                print("Saved default RobertaTokenizer to tokenizer directory")
        except Exception as e:
            print(f"Error setting up tokenizer: {e}")
            return False
    
    print("Model update completed successfully.")
    print("\nTo verify the update, you can run your application and check that it's using the new embeddings.")
    return True

def check_model_compatibility():
    """
    Checks if the model and embeddings are compatible
    """
    model_dir = './model_output/'
    model_path = os.path.join(model_dir, 'roberta_destination_model.pt')
    embeddings_path = os.path.join(model_dir, 'destination_embeddings.npy')
    
    if not os.path.exists(model_path) or not os.path.exists(embeddings_path):
        print("Error: Model or embeddings file not found.")
        return False
    
    try:
        # Load the embeddings to get their shape
        embeddings = np.load(embeddings_path)
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Load the model to check compatibility
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # We need to know how many labels the model was trained with
        # For now, assume it's the number of destinations (first dimension of embeddings)
        num_labels = embeddings.shape[0]
        print(f"Assuming model was trained with {num_labels} labels")
        
        # Initialize model with the current number of labels
        model = DestinationRecommender(num_labels=num_labels).to(device)
        
        # Try to load saved weights
        saved_state = torch.load(model_path, map_location=device)
        model.load_state_dict(saved_state)
        print("Model loaded successfully with current embeddings.")
        
        # Test a simple forward pass
        print("Testing model with random input...")
        input_ids = torch.randint(0, 1000, (1, 512)).to(device)
        attention_mask = torch.ones(1, 512).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            print(f"Model output shape: {outputs.shape}")
        
        print("Model is compatible with current embeddings.")
        return True
        
    except Exception as e:
        print(f"Error checking model compatibility: {e}")
        return False

if __name__ == "__main__":
    print("=== WerTigo Model Updater ===")
    print("This script will update your model with new destination embeddings.\n")
    
    choice = input("Do you want to: \n1. Update embeddings\n2. Check model compatibility\nEnter 1 or 2: ")
    
    if choice == '1':
        if update_model():
            print("\nWould you like to check model compatibility after update? (y/n)")
            if input().lower() == 'y':
                check_model_compatibility()
    elif choice == '2':
        check_model_compatibility()
    else:
        print("Invalid choice. Please run the script again and enter 1 or 2.") 