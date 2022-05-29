# Z-Boson-Minimization

## 2nd Year: Introduction to Programming for Physicists Final Assignment

This python script reads csv files, performs different rounds of validations and then:

- Sorts the data into numerical order and removes any zeros.
- Stacks the two data files to produce one array of data.
- Goes through another round of filtering extreme values.
- Fits the observed data onto a predicted model.
- Minimises the chi squared between the two fits and performs a second minimisation with more outliers removed and a better starting guess for mass and width.

From this, the data containing the mass, m_z, the width, Γ_z, the reduced chi squared and the lifetime can be obtained and printed.
If an error is found in the data input/validation stages, the programme comes to a hault and a statement is printed to the console.

