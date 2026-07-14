# DEFAULT PARAMS

start\_params = {
"lx": 10.0,
"ly": 10.0,
"gap\_size": 10.0,
"prefactor": 1.0,
"delta\_mid\_top": 0.0,
"delta\_mid\_bot": -1.0,
"charges": \[+1.0, -1.0],
"positions": \[np.array(\[6, 5, 1]), np.array(\[3, 2, 1])],
"pw\_error": 1e-8,
}
start\_params\["lz"] = start\_params\["gap\_size"] + 10

end\_params = copy.deepcopy(start\_params)
end\_params\["positions"] = \[np.array(\[6, 5, 9]), np.array(\[3, 2, 9])]



## Changes

* none: flat contrib
* "lx": 20.0,

