###########################
# Problem Set 3b: Space Animals
# Name: Haley Bates-Tarasewicz
# Collaborators: Sam Gordon 
# Time: 2h
#Late Days Used: 1
# Author: charz, brooksjd

from ps3a import load_cows

#================================
# Part B: Transporting Space Chickens
#================================

# Problem 1
def load_chickens(filename):
    """
    Read the contents of the given file.  Assumes the file contents contain
    data in the form of comma-separated chicken name, weight, value triplets, and return a
    dictionary containing chicken names as keys and a tuple of corresponding weight and
    value as values. Assumes all the names are unique.

    Parameters:
    filename - the name of the data file as a string

    Returns:
    a dictionary of chicken name (string), tuple of weight (int) and value (int) pairs
    """
    chickfile = open(filename, 'r') #opens the file as read only 
    chickdict = {} 
    
    for line in chickfile: #loops through each line of the file
        chicklist = line.split(',') #splits each line based on the comma
        chickdict[str(chicklist[0])] = (int(chicklist[1]), int(chicklist[2])) #adds chickens to the dictionary with the value a tuple of value and weight as a key
    
    return chickdict



# Problem 2
def greedy_animal_transport(cows,chickens,limit=20,fuel_constant=5):
    """
    Uses a greedy heuristic to determine an allocation of chickens and cows to
    take in ONE trip that will give the highest value of chickens transported,
    while following the spaceship weight limit and the fuel constraints.
    
    The amount of methane needed to fuel the spaceship for the trip home is the
    weight of the cargo + the spaceship's fuel constant.
    
    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    chickens - a dictionary of name (string), tuple of weight (int) and value
    (int) pairs
    limit - weight limit of the spaceship (an int)
    fuel_constant - the spaceship fuel constant (an int)
    
    Returns:
    A list containing the names of chickens and/or cows transported on the one
    trip
    """
    weight = 0 #starts the weight at 0
    fuel_need = fuel_constant
    fuel = 0
    ret_list = [] #empty list to populate and return 
    
    cow_sort=sorted(cows, key=cows.get, reverse = False) #sorts the list from smallest to largest
    
    new_chicks = {} #new dictionary that lists the chickens based on their value/weight density
    for key in chickens:
        new_chicks[key] = chickens[key][1]*1.0/chickens[key][0]
    
    chicken_sort = sorted(new_chicks, key=new_chicks.get, reverse=True) #sorts the chickens by their density largest to smallest
    
    chicken_test = True
    cow_test = True
    
    while cow_test or chicken_test: #Checks if a chicken was put on the ship, if not, breaks out of the for loop
        chicken_test = False
        cow_test = False
        for x in chicken_sort:
            test_weight = chickens[x][0]
            if weight+test_weight <= limit and fuel_need+test_weight <= fuel: #determines if another chicken will fit on the ship
                ret_list.append(x) #adds the chicken to the trip
                chicken_sort.remove(x)
                weight += test_weight #adds the weight to the ship
                fuel_need += test_weight
                chicken_test = True
                break #breaks out of the for loop
        if not chicken_test: #adds a cow if a chicken has not been added
            for x in cow_sort:
                test_weight = cows[x]
                if weight+test_weight <= limit and (fuel_need+test_weight) <= (fuel+2*test_weight): #determines if another cow will fit on the ship
                    ret_list.append(x) #adds the cow to the trip
                    cow_sort.remove(x)
                    weight+=test_weight #adds the weight to the ship
                    fuel+=2*test_weight #adds cow's fuel production to the ships total fuel
                    fuel_need+=test_weight #adds weight to fuel needs
                    cow_test = True
                    break #gets out of the for loop
                    
        
    return ret_list


# Don't worry about this part; it helps the TAs with testing your code
if __name__ == '__main__':
    chickens = load_chickens('ps3_chicken_data.txt')
    cows = load_cows('ps3_cow_data.txt')
    print cows
    print chickens
    print greedy_animal_transport(cows, chickens)
    
