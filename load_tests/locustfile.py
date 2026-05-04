import random

from locust import HttpUser, between, task

QUERIES = [
    # RAG (internal catalog)
    "What wireless headphones do we have in stock?",
    "Show me AudioMax brand products",
    "Which electronics are rated above 4.5?",
    "Do we have any fitness equipment under $50?",
    # Web search
    "Current market price for noise-cancelling headphones?",
    "Latest reviews for Sony WH-1000XM5",
    "Trending products in home fitness equipment 2024",
    # Pricing / margins
    "Which products have the lowest profit margins?",
    "Calculate average margin for Electronics category",
    "Show me products with margins below 40%",
    # Multi-tool
    "Should we adjust AudioMax headphones pricing vs competitors?",
    "Compare our speaker prices to market rates",
]


class AgentUser(HttpUser):
    wait_time = between(2, 5)

    @task
    def query_agent(self):
        self.client.post(
            "/query",
            json={"query": random.choice(QUERIES)},
            headers={"Content-Type": "application/json"},
        )
