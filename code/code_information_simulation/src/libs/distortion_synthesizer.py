
class SaltAndPepperNoise():
    def __init__(self, prob = None):
        '''
        For salt-and-peppr noises added onto an image of size (h, w),
        each pixel position only have three possible states:
        bright value 255, dark value 0, or the original image value X_{h, w}.
        Therefore, the total amount of values required to encode them is:
        h * w * 3
        Given a specific occurrence probability for the bright and dark values p,
        suppose n values in the representation vector are used to represent one pixel value, then:
        (p / 2) * n values should be used to represent the bright value 255;
        (p / 2) * n values should be used to represent the dark value 0;
        and (1 - p) * n values should be used to represent the original image value X_{h, w}.
        Thus, the total amount of values required to encode the noises then become:
        h * w * n
        '''

        if prob is not None:
            generate vector based on prob
        n = 1
        while True:
            if ((prob / 2) * n).is_integer() and ((1 - prob) * n).is_integer():
                break
            elif n < 1000:
                n += 1
            else:
                raise ValueError("Invalid value for the argument \"prob\" in creating instance of \"SaltAndPepperNoise\".")
        minimum_encoding = img_size[0] * img_size[1] * n
        '''
        the total amount of values that the vector of noise representation can hold:
        length of the vector * number of possible values of each element inside
        '''
        values_in_representation = distort_len * distort_range

        if noise_representation < positional_encoding:
            self.supplement_code = 

        self.
        self.distort_len = distort_len
        self.distort_range = distort_range

    def __call__(self, ):