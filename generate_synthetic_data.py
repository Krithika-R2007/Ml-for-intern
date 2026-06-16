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
    
    # Construct DataFrame
    df = pd.DataFrame({
        'donor_id': donor_ids,
        'previous_donation_amount': previous_donation_amount,
        'donation_frequency': donation_frequency,
        'time_since_last_donation': time_since_last_donation,
        'campaign_participation_history': campaign_participation_history,
        'will_donate_again': will_donate_again
    })
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"Dataset successfully saved to {os.path.abspath(filepath)}")
    print("Class distribution for 'will_donate_again':")
    print(df['will_donate_again'].value_counts(normalize=True))

if __name__ == '__main__':
    generate_data()
