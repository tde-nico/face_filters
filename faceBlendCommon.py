import cv2
import numpy as np
import math

def constrain_point(point, w, h):
	return (
		min(max(point[0], 0),w - 1),
		min(max(point[1], 0),h - 1)
		)


def similarity_transform(in_points, out_points):
	s60 = math.sin(60*math.pi/180)
	c60 = math.cos(60*math.pi/180)
	in_pts = np.copy(in_points).tolist()
	out_pts = np.copy(out_points).tolist()

	xin = c60*(in_pts[0][0] - in_pts[1][0]) - s60*(in_pts[0][1] - in_pts[1][1]) + in_pts[1][0]
	yin = s60*(in_pts[0][0] - in_pts[1][0]) + c60*(in_pts[0][1] - in_pts[1][1]) + in_pts[1][1]
	in_pts.append([np.int(xin), np.int(yin)])

	xout = c60*(out_pts[0][0] - out_pts[1][0]) - s60*(out_pts[0][1] - out_pts[1][1]) + out_pts[1][0]
	yout = s60*(out_pts[0][0] - out_pts[1][0]) + c60*(out_pts[0][1] - out_pts[1][1]) + out_pts[1][1]
	out_pts.append([np.int(xout), np.int(yout)])

	tform = cv2.estimateAffinePartial2D(np.array([in_pts]), np.array([out_pts]))
	return tform[0]


def rect_contains(rect, point):
	if point[0] < rect[0]:
		return 0
	if point[1] < rect[1]:
		return 0
	if point[0] > rect[2]:
		return 0
	if point[1] > rect[3]:
		return 0
	return 1


def calculate_delaunay_triangles(rect, points):
	subdiv = cv2.Subdiv2D(rect)
	for p in points:
		subdiv.insert((int(p[0]), int(p[1])))
	triangle_list = subdiv.getTriangleList()
	delaunay_tri = []

	for t in triangle_list:
		pt = []
		pt.append((t[0], t[1]))
		pt.append((t[2], t[3]))
		pt.append((t[4], t[5]))

		pt1 = (t[0], t[1])
		pt2 = (t[2], t[3])
		pt3 = (t[4], t[5])

		if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
			ind = []
			for j in range(0, 3):
				for k in range(0, len(points)):
					if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
						ind.append(k)
			if len(ind) == 3:
				delaunay_tri.append((ind[0], ind[1], ind[2]))

	return delaunay_tri


def apply_affine_transform(src, src_t, dst_t, size):
	warp_mat = cv2.getAffineTransform(np.float32(src_t), np.float32(dst_t))
	dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None,
		flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
	return dst


def warp_triangle(img1, img2, t1, t2):
	r1 = cv2.boundingRect(np.float32([t1]))
	r2 = cv2.boundingRect(np.float32([t2]))
	t1_rect = []
	t2_rect = []
	t2_rectInt = []

	for i in range(0, 3):
		t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
		t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
		t2_rectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

	mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
	cv2.fillConvexPoly(mask, np.int32(t2_rectInt), (1.0, 1.0, 1.0), 16, 0)

	img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
	size = (r2[2], r2[3])
	img2Rect = apply_affine_transform(img1Rect, t1_rect, t2_rect, size)
	img2Rect = img2Rect * mask

	img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ((1.0, 1.0, 1.0) - mask)
	img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect
