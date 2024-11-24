import numpy as np
import av
import pip
import sys
from av.video.frame import VideoFrame
from math import sqrt, cos, pi, ceil, log, log2, log10
from matplotlib import pyplot as plt
import PIL.Image


class Reformer:
    R = 0.299
    B = 0.114
    G = 0.587

    def RGBtoY(cls, array_list):
        Y_list = []
        for array in array_list:
            new_array = []
        for str_ar in array:
            new_str = [cls.R * RGB[0] + cls.G * RGB[1] + cls.B * RGB[2]
                    for RGB in str_ar]
            new_array += [np.array(new_str)]
            Y_list += [np.array(new_array)]

        return Y_list

    def RGBtoY_str(cls, array_list):
        Y_list = []
        for array in array_list:
            for str_ar in array:
                Y_list += [cls.R * RGB[0] + cls.G * RGB[1] +
                            cls.B * RGB[2] for RGB in str_ar]
        return np.array(Y_list, dtype='float')

    def RGBtoY_input_formatRGB(cls, array_list, frames=None, middle=None):
        new_list = []
        if frames is None:
            frames = len(array_list)
        if middle is None:
            middle = 0
        for array in array_list[middle:middle + frames]:
            new_array = []
            for str_ar in array:
                new_str = [Reformer.getY(RGB[0], RGB[1], RGB[2])
                            for RGB in str_ar]
                new_array += [np.array(new_str)]
                new_list += [np.array(new_array)]
        return new_list

    def YCbCrtoRGB_input_formatYCbCr(cls, array_list, frames=None):
        new_list = []
        if frames is None:
            frames = len(array_list)
            for array in array_list[:frames]:
                new_array = []
                for str_ar in array:
                    new_str = [Reformer.getRGBList(YCbCr[0], YCbCr[1], YCbCr[2])
                               for YCbCr in str_ar]
                    new_array += [np.array(new_str, dtype='uint8')]
                    new_list += [np.array(new_array, dtype='uint8')]

        return new_list

    def getY(cls, R, G, B):
        return int(round(cls.R * R + (1 - cls.R - cls.B) * G + cls.B * B))

    def getCb(cls, R, G, B):
        return int(round(0.5 * (B - Reformer.getY(R, G, B)) / (1 - cls.B)))

    def getCr(cls, R, G, B):
        return int(round(0.5 * (R - Reformer.getY(R, G, B)) / (1 - cls.R)))

    def getR(cls, Y, Cr):
        return Reformer.clip(int(round(Y + (1 - cls.R) / 0.5 * Cr)))

    def getG(cls, Y, Cb, Cr):
        return Reformer.clip(int(round(Y - (2 * cls.B * (1 - cls.B)) /
                                       (1 - cls.B - cls.R) * Cb - (2 * cls.R * (1 - cls.R)) /
                                       (1 - cls.B - cls.R) * Cr)))

    def getB(cls, Y, Cb):
        return Reformer.clip(int(round(Y + (1 - cls.B) / 0.5 * Cb)))

    def getRGBList(cls, Y, Cb, Cr):
        return [Reformer.getR(Y, Cr), Reformer.getG(Y, Cb, Cr), Reformer.getB(Y, Cb)]

    def getYCbCrList(cls, R, G, B):
        return [Reformer.getY(R, G, B), Reformer.getCb(R, G, B), Reformer.getCr(R, G, B)]

    def clip(pixel):
        if pixel > 255:
            return 255
        elif pixel < 0:
            return 0
        return pixel

    def clipAllbyY(array_list):
        clip_array_list = []
        for array in array_list:
            for i in range(0, array.shape[0]):
                for j in range(0, array.shape[1]):
                    array[i][j] = Reformer.clip(array[i][j])
                    clip_array_list += [array]
        return clip_array_list

    def YtoYYY(array_list, frames=None):
        new_list = []
        if frames is None:
            frames = len(array_list)
            for array in array_list[:frames]:
                new_array = []
                for str_ar in array:
                    new_str = [np.array([Reformer.clip(Y), Reformer.clip(
                        Y), Reformer.clip(Y)], dtype='uint8') for Y in str_ar]
                    new_array += [np.array(new_str)]
                    new_list += [np.array(new_array)]
        return new_list

    def retype(array_list, type):
        new_list = []
        for array in array_list:
            array = array.astype(type)
            new_list += [array]
        return new_list

    def ndarray_to_image(array):
        a = array.shape[0]
        for i in range(0, array.shape[0]):
            for j in range(0, array.shape[1]):
                array[i][j][1] = array[i][j][2] = array[i][j][0]
                plt.imshow(array, interpolation='nearest')
        plt.show()

    def frame_to_image(path):
        container = av.open(path)
        array_list = []
        for frame in container.decode(video=0):
            array_list += [frame.to_image()]
        return array_list


class IO:
    last_height = None
    last_width = None
    last_numb_frames = None

    def input_video(path, reformat_name=None):
        array_list = []
        input_container = av.open(path)
        format_container_name = input_container.format.name
        codec_name = input_container.streams.video[0].codec_context.name
        frame_width = input_container.streams.video[0].codec_context.width
        frame_height = input_container.streams.video[0].codec_context.height
        rate = input_container.streams.video[0].average_rate
        format_frame_name = input_container.streams.video[0].codec_context.format.name
        IO.last_height = frame_height
        IO.last_width = frame_width
        IO.last_numb_frames = input_container.streams.video[0].frames
        if reformat_name is None:
            reformat_name = format_frame_name
            for frame in input_container.decode(video=0):
                frame = frame.reformat(
                    frame_width, frame_height, reformat_name)
                array = frame.to_ndarray()
                array_list += [array]
                input_container.close()
        return array_list, format_container_name, reformat_name, codec_name, rate

    def output_video(path, array_list, format_container_name, format_frame_name, codec_name, rate):
        output_container = av.open(
            path, mode='w', format=format_container_name)
        output_stream = output_container.add_stream(codec_name, rate=rate)
        for array in array_list:
            frame = VideoFrame.from_ndarray(array, format=format_frame_name)
            frame = output_stream.encode(frame)
            output_container.mux(frame)
            output_container.close()

    def frame_to_file(array, path):
        with open(path, mode="w") as f:
            for str_ar in array:
                f.write("[")
                for val in str_ar:
                    f.write(f"{val[0]},\t")
                    f.write("]\n")

    def load_image(filename):
        img = PIL.Image.open(filename)
        img.load()
        data = np.asarray(img, dtype="int32")
        return data

    def save_image(npdata, outfilename):
        for i in range(0, npdata.shape[0]):
            for j in range(0, npdata.shape[1]):
                npdata[i][j][0] = Reformer.clip(npdata[i][j][0] + 128)
                npdata[i][j][1] = npdata[i][j][0]
                npdata[i][j][2] = npdata[i][j][0]
                img = PIL.Image.fromarray(np.uint8(npdata)).convert('RGB')
                img.save(outfilename)


class Compress:
    def math_exp(array):
        return array.sum() / array.size

    def sigma(array, math_exp):
        res = 0
        for el in array:
            res += pow(el - math_exp, 2)
        return sqrt(res / array.size)

    def autocorrelation(Y_array, width, height, quantity_frames):
        print(f"autocorrelation start working, data:\nwidth: {
                width}\nheight:{height}\nquantity_frames: {quantity_frames}")
        K = width * height
        offset = - quantity_frames + 1
        list_R = []
        while offset < quantity_frames:
            j = 0
            cur_R = 0
            Y1 = list(Y_array)
            Y2 = list(Y_array)
            if offset > 0:
                del Y1[0:int(K) * offset]
                del Y2[-int(K) * offset:]
            elif offset == 0:
                pass
            else:
                del Y1[int(K) * offset:]
                del Y2[0:-int(K) * offset]
            Y1 = np.array(Y1)
            Y2 = np.array(Y2)
            M_Y1 = Compress.math_exp(Y1)
            M_Y2 = Compress.math_exp(Y2)
            sigma_Y1 = Compress.sigma(Y1, M_Y1)
            sigma_Y2 = Compress.sigma(Y2, M_Y2)
            while j <= K * (quantity_frames - abs(offset)) - 1:
                cur_R += (Y1[j] - M_Y1) * (Y2[j] - M_Y2)
                j += 1
                cur_R /= (K * (quantity_frames - abs(offset))
                          * sigma_Y1 * sigma_Y2)
                list_R += [cur_R]
                print(f"Offset {offset} complete")
                offset += 1
                print("autocorrelation stop working\n")
        return list_R

    def YCbCr(Y_list, N, ratio_step=5, size_count=3):
        result_list = []
        N_max = N
        for array in Y_list:
            ratio = 0
            size = array.shape
            max_str = size[0]
            max_col = size[1]
            count = 0
            if size_count != 0:
                ratio += ratio_step
        while N > 0:
            if count == size_count:
                ratio += ratio_step
                count = 0
                ind_str = N
                ind_col = N - 1
        while ind_col != max_col - (N - 1):
            array[N - 1][ind_col] -= ratio
            array[max_str - (N - 1) - 1][ind_col] -= ratio
            ind_col += 1
        while ind_str != max_str - N:
            array[ind_str][N - 1] -= ratio
            array[ind_str][max_col - (N - 1) - 1] -= ratio
            ind_str += 1
            N -= 1
        if size_count != 0:
            count += 1
            result_list += [array]
            N = N_max
        return result_list

    def toBlocks(array_list, size_block=8):
        heigth, width = array_list[0].shape
        blocks_list = []
        for array in array_list:
            for i in range(0, heigth, size_block):
                for j in range(0, width, size_block):
                    block = np.zeros(
                        (size_block, size_block), dtype="float64")
                    for m in range(i, i + size_block):
                        for n in range(j, j + size_block):
                            block[m - i][n - j] = array[m][n]
                            blocks_list += [block]

        return blocks_list, heigth, width

    def fromBlocks(block_list, height, width, quantity_frames, size_block=8):
        array_list = []
        index = 0
        for fr in range(0, quantity_frames):
            array = np.zeros((height, width), dtype="float64")
            for i in range(0, height, size_block):
                for j in range(0, width, size_block):
                    block = block_list[index]
                    for m in range(i, i + size_block):
                        for n in range(j, j + size_block):
                            array[m][n] = block[m - i][n - j]
                            index += 1
                            array_list += [array]

        return array_list, height, width

    def C(f, N):
        if f == 0:
            return 1. / N
        return 2. / N

    def DCT(X_block, size_block=8):
        Y_block = np.zeros((size_block, size_block), dtype='float64')
        for k in range(0, size_block):
            for l in range(0, size_block):
                sum = 0
                for i in range(0, size_block):
                    for j in range(0, size_block):
                        sum += X_block[i][j] * cos(((2. * i + 1) * pi * k) / (2. * size_block)) * cos(
                            ((2. * j + 1) * pi * l) / (2. * size_block))
                Y_block[k][l] = sqrt(Compress.C(k, size_block)) * \
                    sqrt(Compress.C(l, size_block)) * sum
        return Y_block

    def ODCT(Y_block, size_block=8):
        X_block = np.zeros((size_block, size_block), dtype='float64')
        for i in range(0, size_block):
            for j in range(0, size_block):
                sum = 0
                for k in range(0, size_block):
                    for l in range(0, size_block):
                        sum += sqrt(Compress.C(k, size_block)) * sqrt(
                            Compress.C(l, size_block)) * Y_block[k][l] * cos(((2. * i + 1) * pi * k) / (2. * size_block)) * cos(((2. * j + 1) * pi * l) / (2. * size_block))
                        X_block[i][j] = sum
        return X_block

    def DCT_all_block(X_list, size_block=8):
        return [Compress.DCT(block, size_block=size_block) for block in X_list]

    def ODCT_all_block(Y_list, size_block=8):
        lst = []
        for block in Y_list:
            lst += [Compress.ODCT(block, size_block=size_block)]
        return lst

    def Q(R, size_matrix=8):
        Q_matrix = np.zeros((size_matrix, size_matrix))
        for i in range(0, size_matrix):
            for j in range(0, size_matrix):
                Q_matrix[i][j] = 1 + (i + j) * R
        return Q_matrix

    def quantization(*args, size_block=8):
        Q_block = Compress.Q(args[1], size_block)
        res = np.zeros((size_block, size_block))
        for i in range(0, size_block):
            for j in range(0, size_block):
                res[i][j] = round(args[0][i][j] / Q_block[i][j])
        return res

    def dequantization(*args, size_block=8):
        Q_block = Compress.Q(args[1], size_block)
        res = np.zeros((size_block, size_block))
        for i in range(0, size_block):
            for j in range(0, size_block):
                res[i][j] = args[0][i][j] * Q_block[i][j]
        return res

    def quantization_all(DCT_blocks_list, R, size_block=8):
        return [Compress.quantization(block, R, size_block=size_block) for block in DCT_blocks_list]

    def dequantization_all(quantization_blocks_list, R, size_block=8):
        return [Compress.dequantization(block, R, size_block=size_block) for block in quantization_blocks_list]

    def getDC_array(array_list):
        DC_array = np.zeros(len(array_list), dtype='float64')
        for i in range(0, DC_array.size):
            DC_array[i] = array_list[i][0][0]
        return DC_array

    def getDeltaDC_array(array_list):
        deltaDC_array = np.zeros(len(array_list), dtype='float64')
        mean = 0
        for i in range(0, deltaDC_array.size):
            mean += array_list[i][0][0]
            mean /= deltaDC_array.size
            deltaDC_array[0] = array_list[0][0][0] - mean
        for i in range(1, deltaDC_array.size):
            deltaDC_array[i] = array_list[i][0][0] - array_list[i - 1][0][0]
        return deltaDC_array

    def alg(block):
        AC = np.zeros(block.size - 1, dtype='float64')
        N, x, y, index = block.shape[0], 1, 0, 0
        while x > -1 and y < N:
            while x >= 0 and y <= N - 1:
                AC[index] = block[y][x]
                index += 1
                x -= 1
                y += 1
                x += 1
        while x <= N - 1 and 0 <= y <= N - 1:
            AC[index] = block[y][x]
            index += 1
            x += 1
            y -= 1
            y += 1
            x += 1
            y -= 2
        while not (x == N - 1 and y == N - 1):
            while x <= N - 1 and y >= 0:
                AC[index] = block[y][x]
                index += 1
                x += 1
                y -= 1
                y += 2
                x -= 1
        while x >= 0 and y <= N - 1:
            AC[index] = block[y][x]
            index += 1
            x -= 1
            y += 1
            x += 2
            y -= 1
            AC[index] = block[N - 1][N - 1]
        return AC

    def getAC_all(block_list):
        return [Compress.alg(block) for block in block_list]

    def series(AC_block):
        run, level = [], []
        i = 0
        while i < AC_block.size:
            ro = 0
        while AC_block[i] == 0.:
            ro += 1
            if ro == 16:
                ro -= 1
                break
            else:
                i += 1
                if i >= AC_block.size:
                    i -= 1
                    break
            run += [ro]
            level += [AC_block[i]]
            i += 1
            run += [0]
            level += [0]

        return np.array(run, dtype='float64'), np.array(level, dtype='float64')

    def get_series_run_level_all(AC_list):
        return [Compress.series(AC_block) for AC_block in AC_list]

    def reformat_run_level_from_list_to_concat_array(run_level_list_source):
        run_level_list = run_level_list_source[:]
        run = run_level_list[0][0]
        level = run_level_list[0][1]
        run_level_list.pop(0)
        for element in run_level_list:
            run = np.concatenate([run, element[0]])
            level = np.concatenate([level, element[1]])
        return run, level

    def getBC(blocks_array):
        BC_array = np.zeros(blocks_array.size, dtype='float64')
        for i in range(0, BC_array.size):
            BC_array[i] = ceil(log(abs(blocks_array[i]) + 1))
        return BC_array

    def getBitStream(run_level: tuple, DC: np.ndarray, deltaDC: np.ndarray, deltaDC_BC: np.ndarray, w, h, q_frames, motion_vectors=None):
        BC_level = Compress.getBC(run_level[1])
        BC_dDC = Compress.H(
            Compress.get_array_of_probabilities(deltaDC_BC)) * DC.size
        Magnitude_dDc = np.sum(deltaDC_BC)
        run_BC_level = (Compress.H(
            Compress.get_array_of_probabilities(run_level[0])) + Compress.H(
            Compress.get_array_of_probabilities(BC_level))) * run_level[0].size
        Magnitude_Level = np.sum(BC_level)
        if motion_vectors is not None:
            motion_vectors = np.array(motion_vectors)
            bits_motion_vectors = Compress.H(
                Compress.get_array_of_probabilities(np.array(motion_vectors))) * motion_vectors.size
        else:
            bits_motion_vectors = 0
            sum_bit = BC_dDC + Magnitude_dDc + run_BC_level + \
                Magnitude_Level + bits_motion_vectors
            compr_ratio = w * h * 8 * q_frames / sum_bit
        if bits_motion_vectors != 0:
            info_mv = f"Motion Vectors: {
                bits_motion_vectors / sum_bit * 100} %\n"
        else:
            info_mv = "\n"
            str_inform = f"\nBC(dDC): {BC_dDC / sum_bit * 100} %\nMagnitude(dDC): {Magnitude_dDc / sum_bit * 100} %\n("f"Run, BC("f"Level)): {
                run_BC_level / sum_bit * 100} %\nMagnitude(Level): {Magnitude_Level / sum_bit * 100} %\n{info_mv}\n"
            print(str_inform)

        return sum_bit, compr_ratio, str_inform

    def H(*args):
        H_val = 0
        for i in range(0, args[0][0].size):
            if args[0][0][i] != 0:
                H_val += - args[0][0][i] / args[0][1] * \
                    log2(args[0][0][i] / args[0][1])
        return H_val

    def get_array_of_probabilities(array):
        prob_dict = dict()
        for i in range(0, array.shape[0]):
            if np.array2string(array[i]) in prob_dict:
                prob_dict[np.array2string(array[i])] += 1
            else:
                prob_dict.update({np.array2string(array[i]): 1})
        return np.array(list(prob_dict.values())), array.shape[0]

    def PSNR(array_list, recovered_list_source):
        recovered_list = recovered_list_source[:]
        denominator = 0
        for array in array_list:
            for recover_array in recovered_list:
                for i in range(0, array.shape[0]):
                    for j in range(0, array.shape[1]):
                        denominator += (array[i][j] - recover_array[i][j]) ** 2
                        recovered_list.pop(0)
                        break
        return 10 * (log10(len(array_list)) + log10(array_list[0].shape[0]) + log10(array_list[0].shape[1]) + 2 * log10(255) - log10(denominator))

    def difference_frame(frame_prev, frame_next):
        differ_frame = np.zeros(
            (frame_prev.shape[0], frame_prev.shape[1]), dtype='int32')
        for i in range(0, frame_prev.shape[0]):
            for j in range(0, frame_prev.shape[1]):
                differ_frame[i][j] = frame_prev[i][j] - frame_next[i][j]
        return differ_frame

    def difference_frame_all(array_list):
        it = iter(array_list)
        it2 = iter(array_list)
        differ_array_list = []
        differ_array_list.append(next(it))
        for i in range(0, len(array_list) - 1):
            try:
                differ_array_list.append(
                    Compress.difference_frame(next(it2), next(it)))
            except StopIteration:
                break
        return Reformer.clipAllbyY(differ_array_list)

    def SAD(block_curr, block_prev):
        sad = 0
        for i in range(0, len(block_curr)):
            for j in range(0, len(block_prev)):
                sad += abs(block_curr[i][j] - block_prev[i][j])
                return sad

    def motion_estimation(blocks_list, size_frame_in_block_list, frames_quantity, R=8, size_block=16):
        vectors = []
        for i in range(0, frames_quantity - 1):
            vectors += Compress.td_logarithm_find(
                blocks_list[int(i * size_frame_in_block_list): int((i + 1) *
                                                                   size_frame_in_block_list)],
                blocks_list[int((i + 1) * size_frame_in_block_list): int((i + 2) *
                                                                         size_frame_in_block_list)],
                R,
                size_block
            )

        return vectors

    def get_SAD_for_td_logarithm(current_block, frame_prev, a, b, r):
        indexes = [a * (IO.last_width / 16) + b,
                   a * (IO.last_width / 16) + b + r,
                   a * (IO.last_width / 16) + b - r,
                   (a + r) * (IO.last_width / 16) + b,
                   (a - r) * (IO.last_width / 16) + b]
        searchBlocks_all = [(0, 0), (0 - r, 0), (0 + r, 0),
                            (0, 0 + r), (0, 0 - r)]
        s_center, s_down, s_up, s_left, s_right = sys.maxsize, sys.maxsize, sys.maxsize, sys.maxsize,
        sys.maxsize
        s_center = Compress.SAD(current_block, frame_prev[int(indexes[0])])
        if b + r < IO.last_width / 16:
            s_right = Compress.SAD(current_block, frame_prev[int(indexes[1])])
        if b - abs(r) >= 0:
            s_left = Compress.SAD(current_block, frame_prev[int(indexes[2])])
        if a + r < IO.last_height / 16:
            s_up = Compress.SAD(current_block, frame_prev[int(indexes[3])])
        if a - abs(r) >= 0:
            s_down = Compress.SAD(current_block, frame_prev[int(indexes[4])])
            return s_center, s_down, s_up, s_right, s_left, searchBlocks_all

    def get_SAD(current_block, frame_prev, a, b, delta_a, delta_b):
        if a + delta_a < IO.last_height / 16 and a - abs(
                delta_a) >= 0 and b + delta_b < IO.last_width / 16 and b - abs(delta_b) >= 0:
            return Compress.SAD(current_block, frame_prev[int((a + delta_a) * (IO.last_width / 16) + (b + delta_b))])
        return sys.maxsize

    def td_logarithm_find(frame_prev, frame_cur, R=8, size_block=16):
        a = 0
        b = 0
        res_lst = []
        for current_block in frame_cur:
            r = round(R / 2)
            tmpA = a
            tmpB = b
            while True:
                *SAD_center_down_up_r_l, searchBlocks_all = Compress.get_SAD_for_td_logarithm(
                    current_block, frame_prev, tmpA, tmpB, r)
                ind_min = 0
        for i in range(1, len(SAD_center_down_up_r_l)):
            if SAD_center_down_up_r_l[i] < SAD_center_down_up_r_l[ind_min]:
                ind_min = i
            if ind_min == 0:
                r = round(r / 2)
            else:
                tmpA += searchBlocks_all[ind_min][0]
                tmpB += searchBlocks_all[ind_min][1]
            if r == 1:
                cur_min_SAD = Compress.get_SAD(
                    current_block, frame_prev, tmpA, tmpB, 0, 0)
                resA = tmpA
                resB = tmpB
            for cur_dict in [(-1, -1), (0, -1), (+1, -1), (+1, 0), (+1, +1), (0, +1), (-1, +1), (-1, 0)]:
                if cur_min_SAD > Compress.get_SAD(current_block, frame_prev, tmpA, tmpB,
                                                  cur_dict[0], cur_dict[1]):
                    resA = tmpA
                    resB = tmpB
                    resA += cur_dict[0]
                    resB += cur_dict[1]
                    cur_min_SAD = Compress.get_SAD(current_block, frame_prev, tmpA, tmpB,
                                                   cur_dict[0], cur_dict[1])
                    res_lst.append((resA, resB))
                    break
            b += 1
            if b % (IO.last_width / size_block) == 0:
                a += 1
                b = 0

        return res_lst

    def draw(x: list, y: list, nameTitle: str, nameXlabel: str, nameYlabel: str, nameLabel: str,
             sbplt: bool = False, numberOfSubplot: int = 1, height: int = 2, width: int = 2):
        if sbplt:
            plt.subplot(height, width, numberOfSubplot)
            plt.plot(x, y, label=nameLabel)
            plt.title(nameTitle)
            plt.xlabel(nameXlabel)
            plt.ylabel(nameYlabel)
            plt.grid()
            plt.legend()

    def install(package):
        pip.main(['install', package])


if __name__ == "__main__":
    frames_q = 5
    for flag in [True, False]:
        if flag:
            print('MPEG')
        else:
            print('JPEG')
        # paths = ['../../video/lr1_1.avi',
        #          '../../video/lr1_2.avi', '../../videolr1_3.avi']
        paths = ["lr1_1.avi", "lr1_2.avi", "lr1_3.avi"]
        middle = None
        for path in paths:
            print(f'{path}')
            array_list, format_container_name, format_frame_name, codec_name, rate = IO.input_video(
                path, 'rgb24')
            middle = int(len(array_list) / 2)
            if frames_q is None:
                frames_q = IO.last_numb_frames
            if middle is not None:
                if path is paths[0]:
                    array_list.pop(0)
                else:
                    array_list.pop()
                    array_list = Reformer.RGBtoY_input_formatRGB(
                        array_list, frames_q, middle)
            if flag:
                array_list = Compress.difference_frame_all(array_list)
                frames_q = len(array_list)
                i = 0
                recovered_array_list = Reformer.YtoYYY(array_list)
                recovered_array_list = Reformer.retype(
                    recovered_array_list, 'uint8')
            if flag:
                for array in recovered_array_list:
                    IO.save_image(
                        array, f"{path.removesuffix('.avi')}_DIFF_FRAME_{i}.bmp")
                    i += 1
                    if flag:
                        blocks_list, *size = Compress.toBlocks(array_list, 16)
                        size_frame_in_block_list = IO.last_width * \
                            IO.last_height / (16 * 16)
                        motion_vectors = Compress.motion_estimation(
                            blocks_list, size_frame_in_block_list, frames_q)
                        motion_vectors4 = Compress.motion_estimation(
                            blocks_list, size_frame_in_block_list, frames_q, R=4)
                        motion_vectors16 = Compress.motion_estimation(
                            blocks_list, size_frame_in_block_list, frames_q, R=16)
                        motion_vectors32 = Compress.motion_estimation(
                            blocks_list, size_frame_in_block_list, frames_q, R=32)
                        blocks_list, *size = Compress.toBlocks(array_list)
                        DCT_blocks_list = Compress.DCT_all_block(blocks_list)
                        plt_R = []
                        plt_PSNR = []
                        plt_PSNR4, plt_PSNR16, plt_PSNR32 = [], [], []
                        plt_compression_ratio = []
                        plt_compression_ratio4, plt_compression_ratio16, plt_compression_ratio32 = [], [], []
                    if flag:
                        f = open(f"{path.removesuffix('.avi')
                                    }_MPEG R = 8.txt", mode="w")
                        f4 = open(f"{path.removesuffix('.avi')
                                     }_MPEG R = 4.txt", mode="w")
                        f16 = open(f"{path.removesuffix('.avi')
                                      }_MPEG R = 16.txt", mode="w")
                        f32 = open(f"{path.removesuffix('.avi')
                                      }_MPEG R = 32.txt", mode="w")
                    else:
                        f = open(f"{path.removesuffix(
                            '.avi')}_JPEG.txt", mode="w")
                        for R in [0, 1, 5, 10]:
                            print(f'R = {R}')
                            quants_blocks_list = Compress.quantization_all(
                                DCT_blocks_list, R)
                            DC_array = Compress.getDC_array(quants_blocks_list)
                            dDC_array = Compress.getDeltaDC_array(
                                quants_blocks_list)
                            dDC_BC_array = Compress.getBC(dDC_array)
                            AC_list = Compress.getAC_all(quants_blocks_list)
                            run_level_list = Compress.get_series_run_level_all(
                                AC_list)
                            run_level_tuple_arrays = Compress.reformat_run_level_from_list_to_concat_array(
                                run_level_list)
                    if flag:
                        print('BitStream for R = 4')
                        sum_bit, comp_ratio4, info4 = Compress.getBitStream(run_level_tuple_arrays,
                                                                            DC_array, dDC_array, dDC_BC_array, IO.last_width, IO.last_height,
                                                                            frames_q, motion_vectors4)
                        f4.write(info4)
                    else:
                        sum_bit, comp_ratio, info = Compress.getBitStream(run_level_tuple_arrays, DC_array,
                                                                          dDC_array, dDC_BC_array, IO.last_width, IO.last_height,
                                                                          frames_q)
                        f.write(info)
                        dequant_blocks_list = Compress.dequantization_all(
                            quants_blocks_list, R)
                        ODCT_blocks_list = Compress.ODCT_all_block(
                            dequant_blocks_list)
                        recovered_array_list, *size = Compress.fromBlocks(ODCT_blocks_list, size[0], size[1],
                                                                          frames_q)
                    if flag:
                        plt_PSNR4 += [Compress.PSNR(array_list,
                                                    recovered_array_list)]
                        plt_compression_ratio4 += [comp_ratio4]
                        plt_PSNR += [Compress.PSNR(array_list,
                                                   recovered_array_list)]
                        plt_compression_ratio += [comp_ratio]
                        plt_R += [R]
                        recovered_array_list = Reformer.YtoYYY(
                            recovered_array_list)
                        recovered_array_list = Reformer.retype(
                            recovered_array_list, 'uint8')
                    if not flag:
                        IO.output_video(f'{path.replace(".avi", "")}_OUT_{R}.avi', recovered_array_list,
                                        format_container_name, format_frame_name, codec_name, rate)
                        f.close()
                        f4.close()
                    if flag:
                        path += " (MPEG)"
                    else:
                        path += " (JPEG)"
                    if flag:
                        plt.figure(0)
                        f = open(f"{path.replace('.avi', '')
                                    }dop_Compr(Q); R = 8.txt", mode="w")
                        Compress.draw(plt_R, plt_compression_ratio, "Степень сжатия(R)", "Шаг квантования R",
                                      "Степень сжатия", path + "; R = 8", sbplt=True, numberOfSubplot=1, height=1, width=1)
                        for i in range(0, len(plt_R)):
                            f.write(f'{plt_R[i]} {plt_compression_ratio[i]}\n')
                            f.close()
                            f = open(f"{path.replace('.avi', '')
                                        }dop_Compr(Q); R = 4.txt", mode="w")
                            Compress.draw(plt_R, plt_compression_ratio4, "Степень сжатия(R)", "Шаг квантования R",
                                          "Степень сжатия", path + "; R = 4", sbplt=True,
                                          numberOfSubplot=1, height=1, width=1)
                        for i in range(0, len(plt_R)):
                            f.write(f'{plt_R[i]} {
                                    plt_compression_ratio4[i]}\n')
                            f.close()
                            f = open(f"{path.replace('.avi', '')
                                        }dop_Compr(Q); R = 16.txt", mode="w")
                            Compress.draw(plt_R, plt_compression_ratio16, "Степень сжатия(R)", "Шаг квантования R",
                                          "Степень сжатия", path + "; R = 16", sbplt=True,
                                          numberOfSubplot=1, height=1, width=1)
                        for i in range(0, len(plt_R)):
                            f.write(f'{plt_R[i]} {
                                    plt_compression_ratio16[i]}\n')
                            f.close()
                            f = open(f"{path.replace('.avi', '')
                                        }dop_Compr(Q); R = 32.txt", mode="w")
                            Compress.draw(plt_R, plt_compression_ratio32, "Степень сжатия(R)", "Шаг квантования R",
                                          "Степень сжатия", path + "; R = 32", sbplt=True,
                                          numberOfSubplot=1, height=1, width=1)
                        for i in range(0, len(plt_R)):
                            f.write(f'{plt_R[i]} {
                                    plt_compression_ratio32[i]}\n')
                            f.close()
                            plt.savefig(
                                f'{path.replace(".avi", "")}_DOP_COMPR.png')
                            plt.figure(2)
                            f = open(f"{path.replace('.avi', '')
                                        } PSNR(Q); R = 8.txt", mode="w")
                            Compress.draw(plt_R, plt_PSNR, "PSNR(R)", "Шаг квантования R",
                                          "PSNR", path + "; R = 8", sbplt=True,
                                          numberOfSubplot=1, height=1, width=1)
                        for i in range(0, len(plt_R)):
                            f.write(f'{plt_R[i]} {plt_PSNR[i]}\n')
                            f.close()
                            f = open(f"{path.replace('.avi', '')
                                        } PSNR(Q); R = 4.txt", mode="w")
                            Compress.draw(plt_R, plt_PSNR4, "PSNR(R)", "Шаг квантования R",
                                          "PSNR", path + "; R = 4", sbplt=True,
                                          numberOfSubplot=1, height=1, width=1)
                        for i in range(0, len(plt_R)):
                            f.write(f'{plt_R[i]} {plt_PSNR4[i]}\n')
                            f.close()
                            f = open(f"{path.replace('.avi', '')
                                        } PSNR(Q); R = 16.txt", mode="w")
                            Compress.draw(plt_R, plt_PSNR16, "PSNR(R)", "Шаг квантования R",
                                          "PSNR", path + "; R = 16", sbplt=True,
                                          numberOfSubplot=1, height=1, width=1)
                        for i in range(0, len(plt_R)):
                            f.write(f'{plt_R[i]} {plt_PSNR16[i]}\n')
                            f.close()
                            f = open(f"{path.replace('.avi', '')
                                        } PSNR(Q); R = 32.txt", mode="w")
                            Compress.draw(plt_R, plt_PSNR32, "PSNR(R)", "Шаг квантования R",
                                          "PSNR", path + "; R = 32", sbplt=True,
                                          numberOfSubplot=1, height=1, width=1)
                        for i in range(0, len(plt_R)):
                            f.write(f'{plt_R[i]} {plt_PSNR32[i]}\n')
                            f.close()
                            plt.savefig(
                                f'{path.replace(".avi", "")}_DOP_PSNR.png')
                            plt.figure(3)
                            f = open(
                                f"{path.replace('.avi', '')} Степень сжатия(PSNR); R = 8.txt", mode="w")
                            Compress.draw(plt_PSNR, plt_compression_ratio, "Степень сжатия(PSNR)", "PSNR",
                                          "Степень сжатия", path + "; R = 8", sbplt=True,
                                          numberOfSubplot=1, height=1, width=1)
                        for i in range(0, len(plt_PSNR)):
                            f.write(f'{plt_PSNR[i]} {
                                    plt_compression_ratio[i]}\n')
                            f.close()
                            f = open(
                                f"{path.replace('.avi', '')} Степень сжатия(PSNR); R = 4.txt", mode="w")
                            Compress.draw(plt_PSNR4, plt_compression_ratio4, "Степень сжатия(PSNR)", "PSNR",
                                          "Степень сжатия", path + "; R = 4", sbplt=True,
                                          numberOfSubplot=1, height=1, width=1)
                        for i in range(0, len(plt_PSNR4)):
                            f.write(f'{plt_PSNR4[i]} {
                                    plt_compression_ratio4[i]}\n')
                            f.close()
                            f = open(
                                f"{path.replace('.avi', '')} Степень сжатия(PSNR); R = 16.txt", mode="w")
                            Compress.draw(plt_PSNR16, plt_compression_ratio16, "Степень сжатия(PSNR)", "PSNR",
                                          "Степень сжатия", path + "; R = 16", sbplt=True,
                                          numberOfSubplot=1, height=1, width=1)
                        for i in range(0, len(plt_PSNR16)):
                            f.write(f'{plt_PSNR16[i]} {
                                    plt_compression_ratio16[i]}\n')
                            f.close()
                            f = open(
                                f"{path.replace('.avi', '')} Степень сжатия(PSNR); R = 32.txt", mode="w")
                            Compress.draw(plt_PSNR32, plt_compression_ratio32, "Степень сжатия(PSNR)", "PSNR",
                                          "Степень сжатия", path + "; R = 32", sbplt=True,
                                          numberOfSubplot=1, height=1, width=1)
                        for i in range(0, len(plt_PSNR32)):
                            f.write(f'{plt_PSNR32[i]} {
                                    plt_compression_ratio32[i]}\n')
                            f.close()
                            plt.savefig(
                                f'{path.replace(".avi", "")}_DOP_COMPR_PSNR.png')
                            plt.figure(1)
                            f = open(f"{path.replace('.avi', '')
                                        } PSNR(Q).txt", mode="w")
                            Compress.draw(plt_R, plt_PSNR, "PSNR(R)", "Шаг квантования R", "PSNR", path, sbplt=True,
                                          numberOfSubplot=1)
                        for i in range(0, len(plt_R)):
                            f.write(f'{plt_R[i]} {plt_PSNR[i]}\n')
                            f.close()
                            f = open(f"{path.replace('.avi', '')
                                        }_Compr(PSNR).txt", mode="w")
                            Compress.draw(plt_PSNR, plt_compression_ratio, "Степень сжатия(PSNR)", "PSNR",
                                          "Степень сжатия", path,
                                          sbplt=True,
                                          numberOfSubplot=2)
                        for i in range(0, len(plt_PSNR)):
                            f.write(f'{plt_PSNR[i]} {
                                    plt_compression_ratio[i]}\n')
                            f.close()
                            f = open(f"{path.replace('.avi', '')
                                        }_Compr(Q).txt", mode="w")
                            Compress.draw(plt_R, plt_compression_ratio, "Степень сжатия(R)", "Шаг квантования R",
                                          "Степень сжатия",
                                          path,
                                          sbplt=True,
                                          numberOfSubplot=3)
                        for i in range(0, len(plt_R)):
                            f.write(f'{plt_R[i]} {plt_compression_ratio[i]}\n')
                            f.close()
                            if not flag:
                                plt.savefig(f'{path.removesuffix(".avi")}.png')
                            else:
                                plt.savefig(f'{path.removesuffix(".avi")}.png')
                            if not flag:
                                plt.savefig(f'ALL_GR_JPEG.png')
                            else:
                                plt.savefig(f'ALL_GR_MPEG.png')
                        plt.show()
                        plt.show()
                        plt.show()
