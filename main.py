from .core_processing import *


def main():

    face_1 = 5
    face_2 = 6

    image_1_path = fr'.\FaceImages\Face{face_1}.tif'
    image_2_path = fr'.\FaceImages\Face{face_2}.tif'

    number_points = 12

    image_1 = cv2.imread(image_1_path)
    image_2 = cv2.imread(image_2_path)

    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

    points_1_name = f'Face{face_1}_pts'
    points_2_name = f'Face{face_2}_pts'

    # Saving images points chosen in .npy file
    # get_image_points(image_1, image_2, points_1_name, points_2_name, number_points)

    image_1_pts = np.load(fr'.\Part A\{points_1_name}.npy')
    image_2_pts = np.load(fr'.\Part A\{points_2_name}.npy')

    transformation_type = 1  # 1 = Projective Transformation , 0 = Affine Transformation
    number_frames = 100
    time_points = np.linspace(0, 1, number_frames)

    # morph_seq = create_morph_sequence(image_1, image_1_pts, image_2, image_2_pts, time_points, transformation_type)

    # write_morphing_video(morph_seq, f'Face{face_1}_Face{face_2}_video_Projective')

    # --- Part B ---

    paper_1 = 1
    paper_2 = 2

    image_1_path = fr'.\Paper_Lion_Images\Paper_1.tiff'
    image_2_path = fr'.\Paper_Lion_Images\Paper_2.tiff'

    image_1_lion_path = fr'.\Paper_Lion_Images\Lion_1.jpg'
    image_2_lion_path = fr'.\Paper_Lion_Images\Lion_2.jpg'

    number_points = 12

    image_1_paper = cv2.imread(image_1_path)
    image_2_paper = cv2.imread(image_2_path)

    image_1_lion = cv2.imread(image_1_lion_path)
    image_2_lion = cv2.imread(image_2_lion_path)

    image_1_paper = cv2.cvtColor(image_1_paper, cv2.COLOR_BGR2GRAY)
    image_2_paper = cv2.cvtColor(image_2_paper, cv2.COLOR_BGR2GRAY)

    image_1_lion = cv2.cvtColor(image_1_lion, cv2.COLOR_BGR2GRAY)
    image_2_lion = cv2.cvtColor(image_2_lion, cv2.COLOR_BGR2GRAY)

    points_1_name = f'Paper_{paper_1}_pts'
    points_2_name = f'Paper_{paper_2}_pts'

    # image_1_paper, image_2_paper = image_resizing(image_1, image_2)

    # Resizing the images to a smaller video-handable size
    image_1_paper = cv2.resize(image_1_paper, (600, 600), interpolation=cv2.INTER_CUBIC)
    image_2_paper = cv2.resize(image_2_paper, (600, 600), interpolation=cv2.INTER_CUBIC)

    # image_1_lion = cv2.resize(image_1_railways, (500, 500), interpolation=cv2.INTER_CUBIC)
    # image_2_lion = cv2.resize(image_2_railways, (500, 500), interpolation=cv2.INTER_CUBIC)

    # Saving images points chosen in .npy files
    # get_image_points(image_1_paper, image_2_paper, points_1_name, points_2_name, number_points)
    # get_image_points(image_1_lion, image_2_lion, 'Lion1_pts', 'Lion2_pts', number_points)

    image_1_paper_pts = np.load(fr'.\Part B\{points_1_name}.npy')
    image_2_paper_pts = np.load(fr'.\Part B\{points_1_name}.npy')

    image_1_lion_pts = np.load(fr'.\Part B\Lion1_pts.npy')
    image_2_lion_pts = np.load(fr'.\Part B\Lion2_pts.npy')

    transformation_type = 1  # 1 = Projective Transformation , 0 = Affine Transformation
    number_frames = 100
    time_points = np.linspace(0, 1, number_frames)

    # projective_paper_seq = create_morph_sequence(image_1_paper, image_1_paper_pts, image_2_paper, image_2_paper_pts,
    #                                            time_points, transformation_type)
    # affine_paper_seq = create_morph_sequence(image_1_paper, image_1_paper_pts, image_2_paper, image_2_paper_pts,
    #                                        time_points, 1 - transformation_type)

    # projective_lion_seq = create_morph_sequence(image_1_lion, image_1_lion_pts, image_2_lion,
    #                                           image_2_lion_pts, time_points, transformation_type)
    # affine_lion_seq = create_morph_sequence(image_1_lion, image_1_lion_pts, image_2_lion,
    #                                       image_2_lion_pts, time_points, 1 - transformation_type)

    # write_morphing_video(projective_paper_seq, f'Paper{paper_1}_Paper{paper_2}_video_Projective')
    # write_morphing_video(affine_paper_seq, f'Paper{paper_1}_Paper{paper_2}_video_Affine')

    # write_morphing_video(projective_lion_seq, 'Lion1_Lion2_video_Projective')
    # write_morphing_video(affine_lion_seq, 'Lion1_Lion2_video_Affine')

    # --- Part C - 1 ---

    small_number_points = 5
    large_number_points = 12

    small_points_1_name = f'Face{face_1}_{small_number_points}_pts'
    small_points_2_name = f'Face{face_2}_{small_number_points}_pts'

    large_points_1_name = f'Face{face_1}_{large_number_points}_pts'
    large_points_2_name = f'Face{face_2}_{large_number_points}_pts'

    # get_image_points(image_1, image_2, small_points_1_name, small_points_2_name, small_number_points)

    image_1_small_pts = np.load(fr'.\Part C - 1\{small_points_1_name}.npy')
    image_2_small_pts = np.load(fr'.\Part C - 1\{small_points_2_name}.npy')

    # get_image_points(image_1, image_2, large_points_1_name, large_points_2_name, large_number_points)

    image_1_large_pts = np.load(fr'.\Part C - 1\{large_points_1_name}.npy')
    image_2_large_pts = np.load(fr'.\Part C - 1\{large_points_1_name}.npy')

    seq_small = create_morph_sequence(image_1, image_1_small_pts, image_2, image_2_small_pts, time_points, 0)
    seq_large = create_morph_sequence(image_1, image_1_large_pts, image_2, image_2_large_pts, time_points, 0)
    seq_small_half = seq_small[time_points.shape[0]//2]
    seq_large_half = seq_large[time_points.shape[0]//2]

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(seq_small_half.astype('int'))
    plt.title(f'Image Projective with {small_number_points} of Points')
    plt.subplot(1, 2, 2)
    plt.imshow(seq_large_half.astype('int'))
    plt.title(f'Image Projective with {large_number_points} of Points')
    plt.show()

    # --- Part C - 2 ---

    number_points_sparse = 5

    points_1_non_sparse_name = f'Face{face_1}_non_sparse_pts'
    points_2_non_sparse_name = f'Face{face_2}_non_sparse_pts'

    points_1_sparse_name = f'Face{face_1}_sparse_pts'
    points_2_sparse_name = f'Face{face_2}_sparse_pts'

    # get_image_points(image_1, image_2, points_1_non_sparse_name, points_2_non_sparse_name, number_points_sparse)
    # get_image_points(image_1, image_2, points_1_sparse_name, points_2_sparse_name, number_points_sparse)

    image_1_non_sparse_pts = np.load(fr'.\Part C - 2\{points_1_non_sparse_name}.npy')
    image_2_non_sparse_pts = np.load(fr'.\Part C - 2\{points_2_non_sparse_name}.npy')

    image_1_sparse_pts = np.load(fr'.\Part C - 2\{points_1_sparse_name}.npy')
    image_2_sparse_pts = np.load(fr'.\Part C - 2\{points_1_sparse_name}.npy')

    projective_non_sparse_seq = create_morph_sequence(image_1, image_1_non_sparse_pts,
                                                      image_2, image_2_non_sparse_pts, time_points, 1)
    projective_sparse_seq = create_morph_sequence(image_1, image_1_sparse_pts, image_2,
                                                  image_2_sparse_pts, time_points, 1)
    projective_non_sparse_half = projective_non_sparse_seq[time_points.shape[0]//2]
    projective_sparse_half = projective_sparse_seq[time_points.shape[0]//2]

    plt.figure()
    plt.subplot(1, 4, 1)
    plt.imshow(image_1.astype('int'))
    plt.title("image_1")
    plt.subplot(1, 4, 2)
    plt.imshow(projective_non_sparse_half.astype('int'))
    plt.title("Projective Non Sparse")
    plt.subplot(1, 4, 3)
    plt.imshow(projective_sparse_half.astype('int'))
    plt.title("Projective Sparse")
    plt.subplot(1, 4, 4)
    plt.imshow(image_2.astype('int'))
    plt.title("im2_T")
    plt.show()

    print('\nPass!')


if __name__ == '__main__':
    main()
