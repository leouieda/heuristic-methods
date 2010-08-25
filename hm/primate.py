"""
Generation of prime number sequences and prime number factorization.
"""


def primes(n):
    """
    Returns a list with all the primes up to n
    """

    # Do the sieve of Eratosthenes to find the list of primes
    # Mark the numbers in the sieve with zero if they are out

    sieve = range(n + 1)

    # Get rid of 0 and 1
    sieve[0:2] = [0, 0]

    # When i**2 is greater than n, there are no multiples of i left
    for i in xrange(2, int(n**0.5) + 1):

        if sieve[i] != 0:

            # Remove the multiples of i from sieve
            for j in xrange(i**2, n + 1, i):

                sieve[j] = 0

    # Remove the zeros from the sieve
    sieve = [p for p in sieve if p != 0]

    return sieve



def factor(n, primes):
    """
    Factor n into primes numbers given the list of primes.
    Returns a list with the factors.
    """ 

    factors = [0 for i in xrange(len(primes))]

    res = n

    for i in xrange(len(primes)):

        aux = res % primes[i]

        while aux == 0:

            res = res//primes[i]

            factors[i] += 1

            aux = res % primes[i]

    return factors