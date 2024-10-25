# Django Project with API Integration and Machine Learning

## Overview

This project is a Django application that showcases a variety of talents in web development, API integration, and machine learning. The assignment spans numerous domains, including Django development, API integration, handling massive datasets, and deployment, ensuring that only highly qualified individuals can successfully complete it.

The use of a pre-trained machine learning model enables developers to focus on complicated integrations and logic without requiring extensive ML understanding. This feature assesses a developer's ability to provide advanced functionality in their applications.

The deployment criterion underlines the need of being familiar with production-grade configurations, which experienced developers can efficiently manage. This project is aimed for advanced developers and focuses on:

- Django development
- Advanced API handling
- Database management
- Backtesting logic

## Requirements

To successfully set up and deploy this Django application, please ensure you have the following:

- Python 3.8 or later
- Docker and Docker Compose
- PostgreSQL database (AWS RDS recommended)
- Git for version control

## Setup Instructions

### Local Development Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/doanh280605/blockhouse_backend.git
   cd your_repository
2. **Create a .env file**

3. **Build and run Docker containers:**
    docker-compose up --build

4. **Run migrations and seed data if needed:**
    docker-compose exec web python manage.py migrate

Deployment: 
1. **SSH into your EC2 instance:**

I honestly haven't had any prior experience in CI/CD Pipeline, interacting with amazon RDS and EC2 instance. I had tried to deploy the app with ssh key.pem, but I got access denied and I couldn't fix it. So I failed to deploy the app to a public address. 
Overall this is a great experience for me to realize what I'm missing in the process of learning Software Development, and I really appreciate the opportunity that Blockhouse had given me.

