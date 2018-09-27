## Synopsis

The goal of this project was to apply machine learning principles by building an algorithm to identify Enron Employees who may have committed fraud based on the public Enron financial and email dataset.

## Code Example

```python

#create new features
new_dict = {}
for name in data_dict:

    i = data_dict[name]
    from_poi_to_this_person = i["from_poi_to_this_person"]
    to_messages = i["to_messages"]
    fraction_from_poi = fractionCreater( from_poi_to_this_person, to_messages )
    i["fraction_from_poi"] = fraction_from_poi


    from_this_person_to_poi = i["from_this_person_to_poi"]
    from_messages = i["from_messages"]
    fraction_to_poi = fractionCreater( from_this_person_to_poi, from_messages )
    new_dict[name]={"from_poi_to_this_person":fraction_from_poi,
                       "from_this_person_to_poi":fraction_to_poi}
    i["fraction_to_poi"] = fraction_to_poi
    
features_list = ["poi", "fraction_from_poi", "fraction_to_poi"]    
data = featureFormat(data_dict, features_list)

#plot new features
for point in data:
    from_poi = point[1]
    to_poi = point[2]
    plt.scatter( from_poi, to_poi )
    if point[0] == 1:
        plt.scatter(from_poi, to_poi, color="r", marker="*")
plt.xlabel("from poi(fraction)")
plt.ylabel("to poi(fraction)")
plt.show()

```
