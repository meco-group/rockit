import python_matlab

import rockit

exclude = [r"Ocp\.spy_",r"DirectMethod\.spy_"]
custom = {}
custom[r"Ocp\.spy$"] = f"""
    function out = spy(obj)
       figure;
       subplot(1,2,1);
       [J, titleJ] = obj.jacobian(true);
       spy(casadi.DM(J));
       title(titleJ);
       subplot(1,2,2);
       [H, titleH] = obj.hessian(true);
       spy(casadi.DM(H));
       title(titleH);
    end
""" 
callback_post = {}
callback_pre = {}


python_matlab.generate(rockit,exclude=exclude,custom=custom,callback_post=callback_post,callback_pre=callback_pre)
