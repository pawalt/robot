import unittest
import numpy as np
import torch
from acrobot_swingup.trajectory import TrajectoryController
from acrobot_feasibility.study import design
from .imitation import teacher_action, action_loss


class TeacherTests(unittest.TestCase):
    def test_actuator_equivalent_outputs_and_wrong_sign_gradients(self):
        prediction=torch.tensor([5.,-5.,.3,-2.,2.],requires_grad=True)
        target=torch.tensor([1.,-1.,.3,1.,-1.])
        loss=action_loss(prediction,target,'actuator');loss.backward()
        torch.testing.assert_close(prediction.grad[:3],torch.zeros(3))
        self.assertLess(prediction.grad[3],0)
        self.assertGreater(prediction.grad[4],0)

    def test_vectorized_phases_match_reference_controller(self):
        states=np.array([[0.,0,0,0],[.1,-.2,.3,-.4],[np.pi,0,0,0]])
        actions=np.array([.3,-.5]);gains=np.array([[1.,2,3,4],[-1.,2,-3,4]])
        p=np.eye(4)*100;level=.05
        data=dict(states=states,actions=actions,gains=gains,capture_matrix=p,
                  capture_level=level,balance_gain=design(.05)[0])
        x=np.array([[.01,.02,.03,.04],[.15,-.19,.3,-.4],[np.pi,.001,0,0],[3.,0,.1,0]])
        phases=np.array([0,1,0,2]);caught=np.array([False,False,False,False])
        actual,new_caught=teacher_action(x,phases,data,caught)
        expected=[]
        for state,phase in zip(x,phases):
            controller=TrajectoryController(states,actions,gains,capture_region=dict(lyapunov_matrix=p,empirical_level=level))
            controller.index=int(phase)
            expected.append(controller(state))
        np.testing.assert_allclose(actual,expected)
        np.testing.assert_array_equal(new_caught,[False,False,True,True])
