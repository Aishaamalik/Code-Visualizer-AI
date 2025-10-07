# Testing the functions

def visualize_primes(limit):
    """
    Find and return all prime numbers up to the given limit.
    """
    primes = []
    for num in range(2, limit + 1):
        is_prime = True
        for i in range(2, int(num ** 0.5) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    return primes

# Test the function
limit = 20
prime_numbers = visualize_primes(limit)
print(f"Prime numbers up to {limit}: {prime_numbers}")