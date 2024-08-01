import numpy as np
from general_robotics_toolbox import *
from scipy.optimize import fminbound
import matplotlib.pyplot as plt

def equalize_curve_spacing(curve,num_points,iteration=2):
	###linearly interpolate a given curve with equally spaced num_points points
	#iteration given the case with zigzag redundant motion

	for i in range(iteration):
		lam=calc_lam_cs(curve)
		lam_new=np.linspace(0,lam[-1],num_points)
		curve_new=[curve[0]]
		for i in range(1,num_points):
			idx=np.argsort(np.abs(lam - lam_new[i]))[:2]	###find closest 2 path indices
			weight=(lam_new[i]-lam[idx[0]])/(lam[idx[1]]-lam[idx[0]])	###calc weight between 2 points
			curve_new.append(weight*curve[idx[0]]+(1-weight)*curve[idx[1]])
		curve=np.array(curve_new)
	return curve

def calc_curvature(curve):
	lam=calc_lam_cs(curve)
	dlam=np.gradient(lam)
	curve_tan=np.gradient(curve,axis=0)
	curve_tan_mag = np.linalg.norm(curve_tan,axis=1)
	curve_tan_unit = curve_tan / curve_tan_mag[:, np.newaxis]

	d_curve_tan_unit=np.gradient(curve_tan_unit,axis=0)
	curvature=np.linalg.norm(d_curve_tan_unit,axis=1)/dlam

	return curvature

def calc_lam_js(curve_js,robot):
	#calculate lambda from joints
	lam=[0]
	curve=[]
	for i in range(len(curve_js)):
		robot_pose=robot.fwd(curve_js[i])
		curve.append(robot_pose.p)
		if i>0:
			lam.append(lam[-1]+np.linalg.norm(curve[i]-curve[i-1]))
	return np.array(lam)

def calc_lam_cs(curve):
	###find path length
	temp=np.diff(curve,axis=0)
	temp=np.linalg.norm(temp,axis=1)
	lam=np.insert(np.cumsum(temp),0,0)

	return lam


def find_lmax(lamddot,joint_acc_limit,ddq2dlam2,dqdlam):
	###linesearch for max l_max with lamddot
	return -np.min(np.divide(joint_acc_limit-lamddot*dqdlam,np.abs(ddq2dlam2)))
	
def lamdot_qlambda(robot,curve_js,lam,plot=False):
	###Find the maximum lambda dot for each lambda given qdot and qddot constraints
	joint_acc_limit=robot.get_acc(curve_js)
	################################################Find Lambdadot on qdot constraint##############################################################################################
	###find desired qdot at each step
	dq=np.gradient(curve_js,axis=0)
	dlam=np.gradient(lam)
	dqdlam=np.divide(dq.T,dlam).T
	
	lamdot_max_from_qdot=np.min(np.divide(robot.joint_vel_limit,np.abs(dqdlam)),axis=1)
	################################################Find Lambdadot on qddot constraint##############################################################################################
	lamdot_max_from_qddot = []
	ddq2dlam2=np.divide(np.gradient(dqdlam,axis=0).T,dlam).T
	#search for the maximum l_max through line search of lamddot
	for i in range(len(curve_js)):
		best_lamddot, l_max, _, _ = fminbound(find_lmax,-9999,9999,args=(joint_acc_limit[i],ddq2dlam2[i],dqdlam[i]),full_output=1)
		lamdot_max_from_qddot.append(np.sqrt(-l_max))
	
	lamdot_max = np.minimum(lamdot_max_from_qdot,lamdot_max_from_qddot)

	if plot:
		plt.plot(lam,lamdot_max_from_qdot,label=r'$\dot{\lambda}_{max}$ from $\dot{q}$ constraint')
		plt.plot(lam,lamdot_max_from_qddot,label=r'$\dot{\lambda}_{max}$ from $\ddot{q}$ constraint')
		plt.xlabel(r'$\lambda$ (mm)')
		plt.ylabel(r'$\dot{\lambda}$ (mm/s)')
		plt.title(r'$\dot{\lambda}$ Boundary Profile' )
		plt.legend()
		plt.show()


	return lamdot_max



def lamdot_qlambda_dual(robot1,robot2,curve_js1,curve_js2,lam,plot=False):
	###Find the maximum lambda dot for each lambda given qdot and qddot constraints for dual arm
	joint_vel_limit=np.hstack((robot1.joint_vel_limit,robot2.joint_vel_limit))
	joint_acc_limit=np.hstack((robot1.get_acc(curve_js1),robot2.get_acc(curve_js2)))
	################################################Find Lambdadot on qdot constraint##############################################################################################
	###find desired qdot at each step
	dq=np.gradient(np.hstack((curve_js1,curve_js2)),axis=0)
	dlam=np.gradient(lam)
	dqdlam=np.divide(dq.T,dlam).T
	
	lamdot_max_from_qdot=np.min(np.divide(joint_vel_limit,np.abs(dqdlam)),axis=1)
	################################################Find Lambdadot on qddot constraint##############################################################################################
	lamdot_max_from_qddot = []
	ddq2dlam2=np.divide(np.gradient(dqdlam,axis=0).T,dlam).T
	#search for the maximum l_max through line search of lamddot
	for i in range(len(lam)):
		best_lamddot, l_max, _, _ = fminbound(find_lmax,-9999,9999,args=(joint_acc_limit[i],ddq2dlam2[i],dqdlam[i]),full_output=1)
		lamdot_max_from_qddot.append(np.sqrt(-l_max))
	
	lamdot_max = np.minimum(lamdot_max_from_qdot,lamdot_max_from_qddot)

	if plot:
		plt.plot(lam,lamdot_max_from_qdot,label=r'$\dot{\lambda}_{max}$ from $\dot{q}$ constraint')
		plt.plot(lam,lamdot_max_from_qddot,label=r'$\dot{\lambda}_{max}$ from $\ddot{q}$ constraint')
		plt.xlabel(r'$\lambda$ (mm)')
		plt.ylabel(r'$\dot{\lambda}$ (mm/s)')
		plt.title(r'$\dot{\lambda}$ Boundary Profile' )
		plt.legend()
		plt.show()

	return lamdot_max
		