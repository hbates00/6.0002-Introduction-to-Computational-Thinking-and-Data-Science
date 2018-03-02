###########################
# Problem Set 3a: Space Cows 
# Name: Haley Bates-Tarasewicz
# Collaborators: Sam Gordon
#Late Days Used: 1
# Time: 3hrs

from ps3_partition import get_partitions
import time

#================================
# Part A: Transporting Space Cows
#================================

# Problem 1
def load_cows(filename):
    """
    Read the contents of the given file.  Assumes the file contents contain
    data in the form of comma-separated cow name, weight pairs, and return a
    dictionary containing cow names as keys and corresponding weights as values.

    Parameters:
    filename - the name of the data file as a string

    Returns:
    a dictionary of cow name (string), weight (int) pairs
    """

    cowfile = open(filename, 'r') #opens the file as read only 
    cowdict = {} 
    for line in cowfile: #loops through each line of the file
        cowlist = line.split(',') #splits each line based on the comma
        cowdict[str(cowlist[0])] = int(cowlist[1]) #adds cow to the dictionary with the weight as a key
    
    return cowdict
        

# Problem 2
def greedy_cow_transport(cows,limit=10):
    """
    Uses a greedy heuristic to determine an allocation of cows that minimizes the
    number of spaceship trips needed to transport all the cows.
    The greedy heuristic should follow the following method:

    1. As long as the current trip can fit another cow, add the largest cow that will fit
        to the trip
    2. Once the trip is full, begin a new trip to transport the remaining cows
    
    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int)
    
    Returns:
    A list of lists, with each inner list containing the names of cows
    transported on a particular trip and the overall list containing all the
    trips
    """
    # TODO: Your code here
    fullist = sorted(cows, key=cows.get, reverse=True) #sorts list from largest to Smallest
    trips = []
    count = 0
    
    while len(fullist) > 0: #breaks if there are no more cows to be carried
        trips.append([]) #adds a trip to the list of trips
        totalweight = 0 
        test = True
        
        while test:
            test = False #will break unless a cow is added to the trip
            
            for i in fullist:
                weight = cows[i]
                
                if (totalweight + weight) <= limit:
                    trips[count].append(i) #adds cow to the list
                    fullist.remove(i) #removes cow so it isn't carried twice
                    totalweight += weight #adds cows weight to the total weight
                    test = True #allows the continued adding of cows
                    break #breaks out of the for loop
                else:
                    pass
        count += 1
    
    return trips
    

# Problem 3
def brute_force_cow_transport(cows,limit=10):
    """
    Finds the allocation of cows that minimizes the number of spaceship trips
    via brute force.  The brute force algorithm should follow the following method:

    1. Enumerate all possible ways that the cows can be divided into separate trips
    2. Select the allocation that minimizes the number of trips without making any trip
        that does not obey the weight limitation
    
    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int)
    
    Returns:
    A list of lists, with each inner list containing the names of cows
    transported on a particular trip and the overall list containing all the
    trips
    """
    cowsort = sorted(cows, key=cows.get, reverse=True) #sorts list from largest to Smallest
    
    possible_trips = [] 
    for partitions in get_partitions(cowsort): #runs through the possible journeys and adds them to the possible trips
        possible_trips.append(partitions)
    
    actual_trips = possible_trips[:] #a copy of possible trips to remove things from
    
    for journey in possible_trips: #takes one possible set of trips
        possible = True
        
        for trip in journey: #takes a particular trip in a journey
            tripweight = 0
            
            for cow in trip: #adds the weight of each cow on the trip
                tripweight += cows[cow]
            
            if tripweight > limit:
                possible = False #if the weight of the trip exceeds the limit, the journey is impossible
            else:
                pass
        
        if not possible: #if journey is impossible, it removes the journey from the final list of trips
            actual_trips.remove(journey)
        else:
            pass
    
    best = actual_trips[0] #starts with the best trip being the first in the list
    
    for i in actual_trips: #compares them all and chooses the best list based on length
        if len(i) < len(best):
            best = i
    
    return best
       
# Problem 4
def compare_cow_transport_algorithms():
    """
    Using the data from ps3a_data.txt and the specified weight limit, run your
    greedy_cow_transport and brute_force_cow_transport functions here.
    
    Print out the number of trips returned by each method, and how long each
    method takes to run in seconds.

    Returns:
    Does not return anything.
    """
    
    cows = load_cows('ps3_cow_data.txt')
    
    #greedy cow transport timing 
    print 'Greedy Cow Transport'
    print '---------------'
    startgreed = time.time()
    greed = greedy_cow_transport(cows, limit=10)
    endgreed = time.time()
    print 'Time Taken: ', endgreed - startgreed
    print 'Number of Trips: ', len(greed)

    print
    
    #brute force cow transport timing
    print 'Brute Force Cow Transport'
    print '---------------'
    startbrute = time.time()
    brute = brute_force_cow_transport(cows, limit=10)
    endbrute = time.time()
    print 'Time Taken: ', endbrute - startbrute
    print 'Number of Trips: ', len(brute)


# Don't worry about this part; it helps the TAs with testing your code
if __name__ == '__main__':
    compare_cow_transport_algorithms()
    
