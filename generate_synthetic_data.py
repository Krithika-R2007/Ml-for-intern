import pandas as pd
import numpy as np
import os

def generate_data(num_samples=1000, filepath='donor_data_inr.csv'):
    print(f"Generating synthetic donor dataset in Rupees (INR) with {num_samples} records...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate donor IDs
    donor_ids = [f"D{1000 + i}" for i in range(num_samples)]
    
    # previous_donation_amount: Log-normal distribution to resemble real giving in INR
    # Mean around ₹1000 to ₹1500, range ₹100 to ₹25000
    previous_donation_amount = np.random.lognormal(mean=6.9, sigma=0.9, size=num_samples)
    previous_donation_amount = np.clip(previous_donation_amount, 100.0, 25000.0)
    previous_donation_amount = np.round(previous_donation_amount, 2)
    
    # donation_frequency: how many times they have donated (Poisson distribution centered around 4)
    donation_frequency = np.random.poisson(lam=4, size=num_samples) + 1 # At least 1 donation
    
    # time_since_last_donation: months since last donation (Exponential distribution centered around 8 months)
    time_since_last_donation = np.random.exponential(scale=8.0, size=num_samples)
    time_since_last_donation = np.clip(time_since_last_donation, 1.0, 36.0)
    time_since_last_donation = np.round(time_since_last_donation).astype(int)
    
    # campaign_participation_history: number of campaigns engaged with (binomial distribution)
    campaign_participation_history = np.random.binomial(n=8, p=0.35, size=num_samples)
    
    # Calculate probability of donating again using a logit function to create realistic correlations
    # - Higher frequency: positive impact (+0.25)
    # - Shorter time since last donation: negative impact of months (-0.18)
    # - Higher campaign participation: positive impact (+0.45)
    # - Previous donation amount in INR: slight positive impact (+0.00008)
    logit = (
        -0.5 + 
        0.25 * donation_frequency - 
        0.18 * time_since_last_donation + 
        0.45 * campaign_participation_history + 
        0.00008 * previous_donation_amount
    )
    
    # Add random noise to make the dataset realistic and challenging
    noise = np.random.normal(0, 0.75, size=num_samples)
    logit += noise
    
    # Sigmoid function to convert to probabilities
    probabilities = 1 / (1 + np.exp(-logit))
    
    # Convert probabilities to binary outcomes (1 = Yes, 0 = No)
    will_donate_again = (np.random.rand(num_samples) < probabilities).astype(int)
    
    # Generate preferred channel
    # 0: Email, 1: WhatsApp, 2: SMS, 3: Phone Call, 4: Social Media
    channels = ['Email', 'WhatsApp', 'SMS', 'Phone Call', 'Social Media']
    
    # Normalize inputs for calculating scores
    norm_amount = previous_donation_amount / 5000.0
    norm_freq = donation_frequency / 5.0
    norm_recency = time_since_last_donation / 12.0
    norm_campaigns = campaign_participation_history / 4.0
    
    scores = np.zeros((num_samples, 5))
    scores[:, 0] = 0.5 + 0.4 * norm_freq + 0.3 * norm_amount - 0.2 * norm_recency # Email
    scores[:, 1] = 0.2 + 0.6 * norm_freq + 0.6 * norm_campaigns - 0.3 * norm_recency # WhatsApp
    scores[:, 2] = 0.6 - 0.4 * norm_freq + 0.4 * norm_recency - 0.4 * norm_campaigns # SMS
    scores[:, 3] = -0.5 + 1.2 * norm_amount - 0.1 * norm_recency # Phone Call
    scores[:, 4] = 0.1 + 0.8 * norm_campaigns - 0.3 * norm_freq # Social Media
    
    # Add random noise to make it challenging
    channel_noise = np.random.normal(0, 0.4, size=(num_samples, 5))
    scores += channel_noise
    
    # Take the channel with the highest score
    channel_indices = np.argmax(scores, axis=1)
    preferred_channel = [channels[idx] for idx in channel_indices]
    
    # Construct DataFrame
    df = pd.DataFrame({
        'donor_id': donor_ids,
        'previous_donation_amount': previous_donation_amount,
        'donation_frequency': donation_frequency,
        'time_since_last_donation': time_since_last_donation,
        'campaign_participation_history': campaign_participation_history,
        'will_donate_again': will_donate_again,
        'preferred_channel': preferred_channel
    })
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"Dataset successfully saved to {os.path.abspath(filepath)}")
    print("Class distribution for 'will_donate_again':")
    print(df['will_donate_again'].value_counts(normalize=True))
    print("\nClass distribution for 'preferred_channel':")
    print(df['preferred_channel'].value_counts(normalize=True))

if __name__ == '__main__':
    generate_data()

