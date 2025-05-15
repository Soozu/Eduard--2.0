# Model Update Instructions

This document explains how to update the WerTigo travel recommendation model with new destination embeddings.

## Prerequisites

Before you begin, make sure you have:

1. The new `destination_embeddings.npy` file in your current directory
2. The original `try.py` file containing the `DestinationRecommender` class
3. The existing model at `./model_output/roberta_destination_model.pt`
4. The tokenizer files in the `./tokenizer/` directory

## Update Process

### Step 1: Run the Model Updater

Run the model updater script:

```bash
python model_updater.py
```

The script will:
- Check if all required files exist
- Load and validate the new embeddings
- Back up existing embeddings (if any)
- Update the embeddings file
- Validate that the model works with the new embeddings
- Update metadata if needed

### Step 2: Handle Any Warnings or Prompts

The script might ask for your input in certain situations:

- If the new embeddings have different dimensions than the old ones
- If there's a mismatch between the model's classifier layer size and the number of embeddings
- If there are errors processing the old embeddings

Follow the prompts and decide whether to continue or cancel the update.

### Step 3: Verify the Update

Once the update is complete, run your application to verify that everything works correctly.

If there are issues, the script creates backups of the original files that you can restore:
- `./model_output/destination_embeddings.backup.npy` - Backup of the old embeddings
- `./model_output/roberta_destination_model.backup.pt` - Backup of the original model (if modified)

## Troubleshooting

### Model Size Mismatch

If the model was trained with a different number of destinations than your new embeddings, the script will detect this and offer to create a new model with the correct size. This preserves the model's weights for the existing destinations.

### Tokenizer Issues

If the tokenizer directory is missing, the script will:
1. Create the directory
2. Copy tokenizer files from the current directory if available
3. If necessary, create a default RobertaTokenizer

### Application Crashes

If the application crashes after updating:
1. Check the error messages for clues
2. Try rolling back to the backup files
3. If needed, retrain the model with the new embeddings using `try.py`

## Additional Notes

- The dimensionality of the embeddings (the second dimension) must match between the old and new embeddings.
- The model expects embeddings in the NumPy `.npy` format.
- If the application loads the embeddings directly from the file, make sure the embeddings have the same shape as the old ones to avoid array shape mismatch errors. 