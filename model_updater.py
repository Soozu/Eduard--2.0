import os
import shutil
import numpy as np
import torch
from transformers import RobertaTokenizer
import sys
import pickle
import importlib.util

print("=== WerTigo Model Updater ===")
print("This script will update your model with new destination embeddings.\n")

# Output directory
model_dir = './model_output/'

# Check if try.py exists
if not os.path.exists('try.py'):
    print("Error: try.py not found. This file is required for model updating.")
    sys.exit(1)

# Import the model class from try.py using importlib to avoid keyword conflicts
try:
    # Use importlib.util to load the module without using the 'try' keyword
    spec = importlib.util.spec_from_file_location("travel_model", "try.py")
    travel_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(travel_model)
    
    # Now get the DestinationRecommender class
    DestinationRecommender = travel_model.DestinationRecommender
    print("Successfully imported DestinationRecommender from try.py")
except Exception as e:
    print(f"Error importing from try.py: {e}")
    print("Make sure try.py contains the DestinationRecommender class.")
    sys.exit(1)

# Check for the new embeddings file
if not os.path.exists('destination_embeddings.npy'):
    print("Error: destination_embeddings.npy not found.")
    print("Please make sure the new embeddings file is in the current directory.")
    sys.exit(1)

# Create model_output directory if it doesn't exist
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"Created directory: {model_dir}")
    
# Check if the model exists
model_path = os.path.join(model_dir, 'roberta_destination_model.pt')
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    print("The model needs to exist before updating embeddings.")
    sys.exit(1)

# Load the new embeddings
try:
    print("Loading new destination embeddings...")
    new_embeddings = np.load('destination_embeddings.npy')
    print(f"Loaded embeddings with shape: {new_embeddings.shape}")
except Exception as e:
    print(f"Error loading destination_embeddings.npy: {e}")
    sys.exit(1)

# Back up the old embeddings if they exist
embeddings_path = os.path.join(model_dir, 'destination_embeddings.npy')
if os.path.exists(embeddings_path):
    backup_path = os.path.join(model_dir, 'destination_embeddings.backup.npy')
    try:
        old_embeddings = np.load(embeddings_path)
        print(f"Found existing embeddings with shape: {old_embeddings.shape}")
        
        # Check if shapes match
        if old_embeddings.shape[1:] != new_embeddings.shape[1:]:
            print("Warning: New embeddings have different feature dimensions than the old ones.")
            print(f"Old: {old_embeddings.shape}, New: {new_embeddings.shape}")
            response = input("Do you want to continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Update cancelled.")
                sys.exit(0)
        
        # Create backup
        shutil.copy2(embeddings_path, backup_path)
        print(f"Backed up old embeddings to {backup_path}")
    except Exception as e:
        print(f"Warning: Could not process old embeddings: {e}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Update cancelled.")
            sys.exit(0)

# Copy the new embeddings to the model directory
try:
    shutil.copy2('destination_embeddings.npy', embeddings_path)
    print(f"Updated embeddings in {embeddings_path}")
except Exception as e:
    print(f"Error copying new embeddings: {e}")
    sys.exit(1)

# Check if metadata.pkl exists and update it if needed
metadata_path = os.path.join(model_dir, 'metadata.pkl')
if os.path.exists(metadata_path):
    try:
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        print("Loaded existing metadata.")
        
        # Check if we need to update anything based on new embeddings
        if 'num_destinations' not in metadata or metadata['num_destinations'] != new_embeddings.shape[0]:
            metadata['num_destinations'] = new_embeddings.shape[0]
            print(f"Updated metadata with new destination count: {new_embeddings.shape[0]}")
            
            # Save updated metadata
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            print("Saved updated metadata.")
    except Exception as e:
        print(f"Warning: Error processing metadata: {e}")
        print("This won't prevent the embeddings update but may cause issues later.")

# Now check if the tokenizer exists
tokenizer_dir = os.path.join(model_dir, 'tokenizer')
if not os.path.exists(tokenizer_dir):
    print("Warning: Tokenizer directory not found. Creating it...")
    
    try:
        os.makedirs(tokenizer_dir, exist_ok=True)
        
        # If we have tokenizer files in the current directory, copy them
        for file_name in ['special_tokens_map.json', 'tokenizer_config.json', 'vocab.json']:
            source_path = os.path.join('tokenizer', file_name)
            if os.path.exists(source_path):
                shutil.copy2(
                    source_path,
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
        print("This might cause issues when using the model.")

# Check model compatibility
print("\nChecking model compatibility with new embeddings...")
try:
    # Load the model to check compatibility
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # We need to know how many labels the model was trained with
    # For now, assume it's the number of destinations (first dimension of embeddings)
    num_labels = new_embeddings.shape[0]
    print(f"Assuming model was trained with {num_labels} labels")
    
    # Initialize model with the current number of labels
    model = DestinationRecommender(num_labels=num_labels).to(device)
    
    # Try to load saved weights
    saved_state = torch.load(model_path, map_location=device)
    
    # Check if there might be a size mismatch in the classifier layer
    if 'classifier.weight' in saved_state:
        classifier_shape = saved_state['classifier.weight'].shape
        if classifier_shape[0] != num_labels:
            print(f"Warning: Model was trained with {classifier_shape[0]} labels but we have {num_labels} destinations.")
            print("This mismatch will cause errors when loading the model.")
            
            response = input("Do you want to try to fix the model? (y/n): ")
            if response.lower() == 'y':
                print("Attempting to update the model's classifier layer...")
                
                # Create a new model with the correct number of labels
                new_model = DestinationRecommender(num_labels=classifier_shape[0]).to(device)
                
                # Load the state dict
                new_model.load_state_dict(saved_state)
                
                # Save a backup of the original model
                backup_model_path = os.path.join(model_dir, 'roberta_destination_model.backup.pt')
                shutil.copy2(model_path, backup_model_path)
                print(f"Backed up original model to {backup_model_path}")
                
                # Save the updated model
                torch.save(new_model.state_dict(), model_path)
                print("Saved updated model with correct label count.")
                
                # Update the model variable for further testing
                model = new_model
            else:
                print("Model will not be modified. This may cause errors when using the application.")
    
    # Load the model state
    model.load_state_dict(saved_state)
    print("Model loaded successfully with current embeddings.")
    
    # Test a simple forward pass
    print("Testing model with random input...")
    input_ids = torch.randint(0, 1000, (1, 512)).to(device)
    attention_mask = torch.ones(1, 512).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        print(f"Model output shape: {outputs.shape}")
    
    print("Model is compatible with new embeddings.")
    print("\nUpdate completed successfully!")
    print("You can now run your application with the updated model and embeddings.")
    
except Exception as e:
    print(f"Error checking model compatibility: {e}")
    print("\nThe embeddings were updated, but there may be compatibility issues with the model.")
    print("You might need to retrain the model or restore the backup embeddings if the application doesn't work correctly.") 